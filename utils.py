import json, pandas as pd, os, numpy as np, pickle, gzip, base64, io, yaml
from itertools import groupby
from collections import Counter
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from IPython.display import display_javascript, display_html, display
from PIL import Image, ImageOps
from pprint import pprint
from copy import deepcopy
from random import Random
from types import GeneratorType as generator
import sqlite3

from fastai.vision.all import *
from fastcore.basics import AttrDict
from fastprogress.fastprogress import progress_bar
import torchvision, torch
from torch import Tensor
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

try: import kornia
except: pass

#from bunch import Bunch
Bunch = AttrDict

def bunchize(x):
    if isinstance(x, list):
        return [ bunchize(y) for y in x ]
    if isinstance(x, dict):
        return AttrDict({ k:bunchize(v) for k,v in x.items() })
    else:
        return x

def mapv(*args, **kwargs):
    "Like map() but returns a list."
    return list(map(*args, **kwargs))

def dict_slice(d, start=0, end=-1):
    return dict(list(d.items())[start:end])

def chunker(iterable, n):
    while True:
        x = next(iterable, None)
        if type(x) != NoneType:
            yield [x] + list(itertools.islice(iterable, n-1))
        else:
            break

def try_(f):
    try:
        return f()
    except:
        return None

def tryf(f):
    def f2(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return None
    return f2

def identity(x):
    return x

def listdir(path):
    for f in os.listdir(path):
        yield os.path.join(path, f)

def to_image(t):
    return (t*255.).to(torch.uint8).permute(1,2,0)

def batch_to_imgs(b):
    b = torch.clamp(b, 0, 1)
    b = b * 255
    return b.permute((0, 2,3,1)).detach().cpu().numpy().astype(np.uint8)

def imgs_to_batch(fs):
    b = torch.stack([ Tensor(np.array(Image.open(f))) for f in fs ])
    return b.permute(0, 3,1,2).float()/255.0

def open_image(fname):
    return imgs_to_batch([fname])[0]

def path_to_thumb(f, dir='pics_thumbs_216x384'):
    n = os.path.basename(f)
    return f'{dir}/{n}.png' # double extension yeah

def path_to_head_thumb(f):
    return path_to_thumb(f, dir='pics_heads_128x128')

def im_to_gallery(ims, cols = 5, figsize = (15,5), titles=None, cmap=None):
    rows = math.ceil(len(ims) / cols)
    for row, ims_ in enumerate(chunked(iter(ims), cols)):
        fig, axs = plt.subplots(1, cols, figsize=figsize)
        for col, im in enumerate(ims_):
            i = row*cols+col
            if cols == 1: ax = axs
            else: ax = axs[col]
            ax.axis('off')
            if titles:
                ax.set_title(titles[i])
            ax.imshow(im, aspect='equal', cmap=cmap)
        plt.axis('off')
        plt.show()

def path_to_gallery(fs, **kwargs):
    im_to_gallery([ Image.open(f) for f in fs ], **kwargs)

def path_to_autocrop(path):
    im = Image.open( path )
    return im.crop( im.getbbox() )

def recur_f(f, x):
    type_ = type(x)
    if type_ == list or type_ == tuple:
        return type_( recur_f(f, y) for y in x )
    elif isinstance(x, Tensor):
        return f(x)
    raise Exception(f"can't recur into unknown type: {type_}")

def recur_shape(x):
    return recur_f(lambda t: t.shape, x)

def NamedSequential(**kwargs):
    return nn.Sequential(OrderedDict([ (n,l) for n,l in kwargs.items() ]))

class ApplyF(Module):
    def __init__(self, f):
        self.f = f
    def forward(self, x):
        return self.f(x)
    def extra_repr(self):
        return f"{self.f.__name__}"

def Reshape(*shape):
    return ApplyF(lambda x: x.reshape(*shape))
def Crop(top, left, height, width):
    return ApplyF(lambda x: transforms.functional.crop(x, top, left, height, width))

class Parallel(nn.Sequential):
    def __init__(self, reducer, blocks):
        nn.Sequential.__init__(self, *blocks)
        self.reducer = reducer

    def forward(self, x: Tensor) -> Tensor:
        ys = [ b(x) for b in self ]
        return self.reducer(*ys)

    def extra_repr(self):
        return f"reducer={self.reducer}, {nn.Sequential.extra_repr(self)}"

def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight)
    elif isinstance(model, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)
    for l in model.children():
        init_weights(l)

def conv1x1(featin, featout, stride=1):
    return nn.Conv2d(featin, featout, kernel_size=1, bias=False, stride=stride)
def conv3x3(featin, featout, stride=1):
    return nn.Conv2d(featin, featout, kernel_size=3, padding=1, bias=False, stride=stride)

# coordconv
def make_coords_feats(shape):
    (w, h) = shape
    wfs = (2*torch.arange(end=h).float()/h - 1).repeat((w, 1))
    hfs = (2*torch.arange(end=w).float()/w - 1)[:,None].repeat((1, h))
    radfs = (wfs**2 * hfs**2).sqrt()
    feats = torch.stack([wfs, hfs, radfs])
    return feats

def im_xmp_json(im):
    if im.format == "PNG":
        desc = im.text.get('Description', None)
        if desc is None:
            return
        s1 = 'lfs|json1|'
        p1 = desc.find(s1)
        if p1 < 0:
            return
        return json.loads(desc[p1+len(s1):])
    elif im.format == "WEBP":
        xmp = im.info.get('xmp')
        if xmp is None:
            return
        xmp = xmp.decode('utf8')
        s1 = '<exif:UserComment>lfs|json' # with version: lfs|json1|
        p1 = xmp.find(s1)
        p2 = xmp.find('</exif:UserComment>')
        if p1 < 0 or p2 < 0:
            return
        assert xmp[p1+len(s1)  ] in ('1', '2'), "supported version"
        assert xmp[p1+len(s1)+1] == '|', "tag end"
        return json.loads(xmp[p1+len(s1)+2 : p2])
    else:
        raise Exception("unsupported format")

def get_in(m, ks, default=None):
    if type(ks) == str:
        ks = ks.split('.')
    while len(ks) > 0:
        k, *ks = ks
        m = m.get(k, None)
        if m is None:
            return default
    return m

# only latest metadata includes avatar id :( oops
def makeup_avatar_id(m):
    a = m.targetPlayer.avatar
    return f'{a.authorId}:{a.name}'


def make_ms(fs, cache_file="py_dataset_ms.json.gz"):
    facing = Tensor([0, 0, -1]).float()
    def go(f):
        im = Image.open(f)
        if im.mode != 'RGBA': return
        m = im_xmp_json(im)
        if not m: return
        m = bunchize(m)
        extra = m.get('extra', None)
        if extra is None: return
        targetPlayerId = extra.get('targetPlayerId', None)
        if targetPlayerId is None: return
        if m.extra.get('pedestalWorldMode', False): return
        targetPlayer = m.extra.get('targetPlayer', None)
        if targetPlayer is None:
            try:
                m.targetPlayer = next(p for p in m.visiblePlayers if p.id == targetPlayerId)
            except StopIteration:
                return
        else:
            m.targetPlayer = targetPlayer
        #if torch.dot(facing, tensor(m.targetPlayer.facing)) < 0.9: return # Only facing direction (front)
        m.targetPlayer.avatar.id = makeup_avatar_id(m)
        del m.visiblePlayers # bloated, not useful
        return Bunch(f=f, fdir=os.path.dirname(f),
                im=Bunch(size=list(im.size), mode=im.mode),
                **bunchize(m))

    def cached_go(f, cache):
        if f in cache:
            return bunchize(cache[f])
        else:
            m = go(f)
            cache[f] = m
            return m

    # we remove the ones without meta, 'extra/targetPlayerId', or missing it in visiblePlayers
    if os.path.exists(cache_file):
        with gzip.open(cache_file, 'r') as fd:
            cache = json.load(fd)
    else:
        cache = {}
    ms = list(filter(identity, [
        cached_go(f, cache)
        for f in with_progressbar(fs) ]))
    with gzip.open(cache_file, 'wt') as fd:
        json.dump(cache, fd)
    return ms


def load_ms(cache_file="py_dataset_ms.json.gz"):
    with gzip.open(cache_file, 'r') as fd:
        cache = json.load(fd)
    return [ bunchize(m) for f, m in cache.items() if type(m) == dict ]


def ms_sort_by_facing(ms, facing=Tensor([0, 0, -1])):
    facings = Tensor([ m.targetPlayer.facing for m in ms ])
    idxs = torch.argsort(-(facing[None,:] * facings).sum(dim=1))
    return [ ms[idx.item()] for idx in idxs ]

def ms_to_most_facing(ms, facing=Tensor([0, 0, -1])):
    # find the best match based on facing
    facings = Tensor([ m.targetPlayer.facing for m in ms ])
    idx = torch.argmax((facing[None,:] * facings).sum(dim=1))
    return ms[idx]

def query_embeds_to_idxs(embeds, e, limit=10, unique_avatar=False, ms=None, skip_first=False, skip_aid=None):
    # cosine similarity
    sims = (e * embeds).sum(dim=1) / (e.norm() * embeds.norm(dim=1))
    idxs = torch.argsort(sims, descending=True)
    sims = sims[idxs]
    if unique_avatar or skip_aid:
        assert ms != None
        def is_seen(idx, aid, seen_aids = set()):
            v = aid in seen_aids
            seen_aids.add(aid)
            return v
        def is_aid(idx, aid):
            return aid == skip_aid
        def should_skip(idx):
            v = False
            aid = ms[idx].targetPlayer.avatar.id
            if unique_avatar:
                v = v or is_seen(idx, aid)
            if skip_aid:
                v = v or is_aid(idx, aid)
            return v
        sims, idxs = zip(*[ (sim, idx)
            for sim, idx in zip(sims, idxs)
            if not should_skip(idx) ])
    starti = 1 if skip_first else 0
    return idxs[starti:limit+starti], sims[starti:limit+starti]

def ann_to_idxs(ann_attr, value, ms, msd, aid_to_anns):
    return [ ms.index( ms_to_most_facing( msd[ aid ] ) )
                  for aid, anns in aid_to_anns.items()
                  if value == anns.get( ann_attr, '' ) ]

def tag_to_idxs(tag, ms, msd, aid_to_anns):
    return [ ms.index( ms_to_most_facing( msd[ aid ] ) )
                  for aid, anns in aid_to_anns.items()
                  if tag in anns.get( 'tags', [] ) ]

def category_to_idxs(category, ms, msd, aid_to_anns):
    return [ ms.index( ms_to_most_facing( msd[ aid ] ) )
                  for aid, anns in aid_to_anns.items()
                  if category == anns.get( 'category', None ) ]

class PILImageRGBA(PILImage):
    _open_args = {'mode': 'RGBA'}

def rand_translate(t, magnitude=25):
    ts = torch.randint(-magnitude, magnitude, (t.shape[0], 2)).float().to(t.device)
    return kornia.geometry.transform.translate(t, ts)

def rand_gaussian(t, kernel_size=7, sigma=5, p=0.5):
    k = kernel_size
    if random.random() > p:
        return t
    s = random.random() * sigma
    return kornia.filters.gaussian_blur2d(t, kernel_size=(k,k), sigma=(s,s))

def make_kornia_augment(kornia_augment_color, kornia_augment_geo):
    # we have RGBA/GA which kornia color augm doesn't support
    # so let's isolate A, pass RGB through kornia chans and recompose RGBA
    def go(x):
        if len(x.shape) != 4: return x
        colors = x.shape[1]-1
        assert colors == 1 or colors == 3, '(grayscale or rgb) + alpha'
        c = x[:,:colors,:,:]
        alpha = x[:,colors,None,:,:]
        x = torch.concat([
            # ensure bg is pure black, always
            alpha * kornia_augment_color(c),
            alpha],
        dim=1)
        return kornia_augment_geo(x)
    return go

format_shape = lambda y: int(reduce(lambda a,b: a*b, y.shape[1:])/1000.)

def print_flow_recur(l, t, levels=None):
    if type(l) in [nn.Sequential, NamedSequential]:
        for i,l in enumerate(l):
            t = print_flow_recur(l, t, levels=(levels or []) + [i] )
    else:
        with torch.no_grad():
            t = l(t)
        levelstr = ':'.join(map(str, levels))
        print(f'layer {levelstr} {t.shape} ({format_shape(t)}K) <- {l.__class__.__name__}')
    return t

def print_flow(m, t):
    m.eval()
    print(f'initial shape {t.shape} ({format_shape(t)}K)')
    print_flow_recur(m, t)

def compute_embeds(fs_vert, encoder, bs=64):
    encoder.eval()
    embeds = []
    pb = progress_bar(range(len(fs_vert)))
    for i, fs in enumerate(chunker(iter(fs_vert), bs)):
        t = imgs_to_batch(fs)
        with torch.no_grad():
            embeds.extend(list( encoder(t.cuda()).cpu() ))
        pb.update(i*len(fs))
    return torch.stack(embeds)

def reco_from_embeds(fs_vert, embeds, tpath):
    idx = fs_vert.index(tpath)
    e = embeds[idx]
    sims = (e * embeds).sum(dim=1) / (e.norm() * embeds.norm(dim=1))
    idxs = torch.argsort(sims, descending=True)
    print(idx, sims.shape, idxs.shape, sims[idx], sims[0], idxs[0:10])
    path_to_gallery(fs=    [ fs_vert[j] for j in idxs[0:20] ],
                    titles=[ f'{sims[j].item():.3f}' for j in idxs[0:20] ],
                    figsize=(10,15))

def batch_dot(t1, t2):
    return (t1*t2).sum(dim=-1)/(t1.norm(dim=-1)*t2.norm(dim=-1))

def tensors_interleave(ts):
    return torch.stack(ts).T.contiguous().view(-1)

def im_to_base64(pil_im, format='png', quality=85):
    b = io.BytesIO()
    pil_im.save(b, format, quality=quality)
    return base64.b64encode(b.getvalue()).decode('ascii')

def select_keys(m, ks): # rebuild to follow ks order
    return { k: m[k] for k in ks if k in m }

def filter_keys(m, ks): # preserve original map order
    ks = set(ks)
    return { k:v for k,v in m.items() if k in ks }

def remove_keys(m, ks):
    ks = set(ks)
    return { k:v for k,v in m.items() if not k in ks }

def safe_removes(xs, to_removes):
    "Remove provided elements from xs in place; doesn't fail if missing."
    for x in to_removes:
        try:
            xs.remove(x)
        except ValueError:
            pass
    return xs

def with_progressbar(xs, n=None):
    if n is None:
        if type(xs) is generator:
            xs = list(xs)
        n = len(xs)
    pb = progress_bar(range(n))
    pb.update(0)
    for i, x in enumerate(xs):
        pb.update(i)
        yield x

# source: https://forums.fast.ai/t/plotting-metrics-after-learning/69937/2
@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    plt.show()

def merge_dicts(dicts):
    return reduce(lambda acc, x: {**acc, **x}, dicts, {})

def merge_anns(anns1, anns2):
    collisions = set( anns1.keys() ).intersection( anns2.keys() ).difference([ 'name', 'tags' ])
    assert len(collisions) == 0, f'collision anns: {collisions} between {anns1} and {anns2}'
    # merge tags lists (deduplicate)
    tags = list(set( anns1.get('tags', []) + anns2.get('tags', []) ))
    return Bunch(merge_dicts([anns1, anns2, ({'tags': tags} if len(tags) > 0 else {})]))

def dict_nonull(**kwargs):
    return { k:v for k,v in kwargs.items() if v != None }

def jsonl_dump(xs, fd):
    for x in xs:
        fd.write((json.dumps(x) + '\n').encode('utf8'))

def jsonl_load(fd):
    for l in fd:
        yield json.loads(l)

class SQLiteDB(object):

    def __init__(self, path, name, col_types):
        '''assumes first column in col_types is the unique id'''
        self.path = path
        self.name = name

        self.execute('pragma journal_mode = WAL')
        self.execute('pragma mmap_size = 30000000000')

        self._create_table(col_types)
        self.cols = list(zip(*col_types))[0]

        self.insert_query = 'INSERT OR IGNORE INTO %s VALUES (%s)' % (self.name, ', '.join(['?'] * len(self.cols)))
        self.update_query = 'UPDATE %s SET %s WHERE %s = ?' % (self.name, self.unique_col, ', '.join(['%s = ?' % col for col in self.cols[1:]]))

        self._create_index()

    def _create_index(self):
        self.execute('CREATE INDEX IF NOT EXISTS %s_index on %s (%s)' % (self.name, self.name, self.unique_col))

    def _create_table(self, col_types):
        self.execute('CREATE TABLE IF NOT EXISTS %s (%s)' % (self.name, ', '.join(['%s %s %s' % (c, t, 'UNIQUE' if i == 0 else '') for i, (c, t) in enumerate(col_types)])))

    @property
    def unique_col(self):
        return self.cols[0]

    def _execute(self, *args, chunk_size=1000):
        with sqlite3.connect(self.path, timeout=1000) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            if type(args[-1]) == types.GeneratorType:
                cur.executemany(*args)
            else:
                cur.execute(*args)
            con.commit()
            while True:
                results = cur.fetchmany(chunk_size)
                if len(results) == 0:
                    break
                for result in results:
                    yield result

    def execute(self, *args, generator=False, **kwargs):
        results = self._execute(*args, **kwargs)
        if not generator:
            results = list(results)
        return results

    def safely_execute(self, *args):
        try:
            self.execute(*args)
            return True
        except sqlite3.IntegrityError:
            return False

    def get(self, rowid):
        results = self.execute('SELECT * FROM %s WHERE rowid = ?' % self.name, (rowid,))
        return Bunch(results[0])

    def search(self, text, col):
        results = self.execute('SELECT * from %s WHERE %s MATCH ?' % (self.name, col), (text,))
        for result in results:
            yield(Bunch(result))

    def size(self):
        results = self.execute('SELECT COUNT(*) FROM %s' % self.name)
        return results[0][0]

    def iter_rows(self):
        rows = self.execute('SELECT * FROM %s' % self.name, generator=True)
        for row in rows:
            yield(Bunch(row))

    def contains(self, elt):
        results = self.execute('SELECT COUNT(*) FROM %s WHERE %s = ?' % (self.name, self.unique_col), (elt[self.unique_col],))
        count = results[0][0]
        return count == 1

    def _iter_insertable_elts(self, elts):
        for elt in elts:
            yield(tuple([elt.get(col, None) for col in self.cols]))

    def insert(self, elts):
        return self.safely_execute(self.insert_query, self._iter_insertable_elts(elts))

    def update(self, elt):
        return self.safely_execute(self.update_query, tuple([elt.get(col, None) for col in self.cols[1:]] + [elt[self.unique_col]]))
