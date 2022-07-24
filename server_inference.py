#!/usr/bin/env python
# coding: utf-8

from utils import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


# In[2]:


# grayscale only
main_dir = '../pics_heads_224x224'
#main_dir = '../pics_thumbs_v3-normalized-grayscaled_216x384'

def path_to_thumb(f, dir=main_dir):
    n = os.path.basename(f)
    return f'{dir}/{n}.png' # double extension yeah

def imgs_to_batch(fs):
    return ims_to_batch(map(image_open, fs))

def image_open(f):
    with Image.open(f) as im:
        #return im.getchannel('L') # for heads
        return im.getchannel('R') # for body

def ims_to_batch(ims):
    b = torch.stack([ Tensor( np.array( im ) ) for im in ims ])
    b = b.unsqueeze(3)
    return b.permute(0, 3,1,2).float()/255.0

def path_to_gallery(fs, **kwargs):
    im_to_gallery([ image_open(f) for f in fs ], **kwargs)

def batch_to_imgs(b):
    b = torch.clamp(b, 0, 1) * 255
    return b.permute((0, 2,3,1)).detach().cpu().numpy().astype(np.uint8)


# In[3]:


model_name = 'model_vrc-heads_similarity3_clip_v2-grayscale-50epoches'
#model_name = 'model_vrc_similarity3_clip_v2-grayscale-75epoches'


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print('device', device)
x = torch.load( f'database_aid-embeds_{model_name}', map_location=torch.device(device) )
embeds = x['embeds']
aids = x['aids']
assert len(embeds) == len(aids)
print('embeds aids', embeds.shape, len(aids))
del x


# In[6]:


# avatar id -> m (info dict from ms)
def go(ms0):
    msd = defaultdict(list)
    no_thumbs = 0
    wrong_res = 0

    for m in ms0:
        #fthumb = path_to_thumb( m.f )
        #if not os.path.exists( fthumb ) or os.path.getsize( fthumb ) <= 0:
        #    no_thumbs += 1
        #    continue
        if m.im.size != [2160, 3840]:
            wrong_res += 1
            continue
        tPlayer = m.targetPlayer
        aid = tPlayer.avatar.id
        msd[aid].append(m)
    cnt_ = len(msd)
    print('cnt_', cnt_)
    print('no_thumbs', no_thumbs)
    print('wrong_res', wrong_res)

    msd2 = { aid:msd[aid] for aid in aids }
    ms = [ m for aid,ms_ in msd2.items() for m in ms_ ]
    return msd2, ms

msd, ms = go(load_ms())
fs = [ m.f for m in ms ]
msd_aids = list(msd.keys())

len(set([ m.targetPlayer.avatar.id for m in ms ])), len(msd), len(ms), list(msd.keys())[0:10]


# In[9]:


with open('annotations_avatar_name_info.yaml', 'r') as fd:
    avatar_to_anns = { name: Bunch(name=name, **anns) for name,anns in yaml.safe_load(fd).items() }

with open('annotations_aid_avatars.yaml', 'r') as fd:
    aid_to_anns = { aid: Bunch(avatar_id=aid, **merge_anns(anns, avatar_to_anns.get(anns.get('name', None), {})))
                   for aid, anns in yaml.safe_load(fd).items()
                   }

len(avatar_to_anns), next(iter(avatar_to_anns.items())), len(aid_to_anns), next(iter(aid_to_anns.items()))


# In[10]:


import clip
clip_model, preprocess = clip.load("RN50", device=device)
clip_model.eval()
device, clip_model


# In[11]:


clip_featsize = 1024
featsizes = (clip_featsize,)

encoder = NamedSequential(
    preprocess=nn.Sequential(
        ApplyF(lambda t: torch.cat([ t, t, t ], dim=1)) # grayscale -> RGB for clip
    ),
    clip=ApplyF(torch.no_grad()(lambda t: clip_model.encode_image( t ).float())),
    projector=nn.Sequential(
        nn.Linear(clip_featsize, featsizes[-1]),
    )
).eval().to(device=device)
encoder.load_state_dict( torch.load(model_name, map_location=torch.device(device)) ), encoder


# In[12]:



# In[13]:


# In[18]:


def query(embeds, e, limit=5, **kwargs):
    def go(idx, sim):
        aid = msd_aids[idx]
        anns = aid_to_anns.get( aid, {} )
        return Bunch(select_keys(locals(), 'idx sim aid anns'.split()))

    idxs, sims = query_embeds_to_idxs( embeds, e, limit=limit, **kwargs)
    return [ go(idx.item(), sim.item()) for idx, sim in zip(idxs, sims) ]

def query_show(embeds, e, limit=14, **kwargs):
    rs = query(embeds, e, limit, **kwargs)
    print('idxs', [ r.idx for r in rs ])
    path_to_gallery(fs=    [ path_to_thumb( ms_to_most_facing( msd[r.aid] ).f ) for r in rs ],
                    titles=[ f'{r.sim:.3f} (#{r.idx})' for r in rs ],
                    figsize=(17,20), cols=7, cmap='gray',
                   )
    return rs


# In[19]:


app = FastAPI()


@app.get("/")
def get_root():
    return "Hi, this is aviCLIP"



# source: https://stackoverflow.com/a/7170023
import operator
def equalize(im):
    h = im.histogram()
    lut = []
    for b in range(0, len(h), 256):
        # step size
        step = reduce(operator.add, h[b:b+256]) / 200 # not max out (255)
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]
    # map image through lookup table
    return im.point(lut)


input_size = 224

async def process_image(image):
    rawimage = io.BytesIO(await image.read())
    im = Image.open(rawimage)
    print(f'Query with image: mode={im.mode} size={im.size} filename={image.filename}')

    w, h = im.size
    if w < 0.8*input_size or h < 0.8*input_size:
        raise Exception('image too small')

    im = ImageOps.grayscale(im)

    aratio = w / h
    if aratio > 1.5 or aratio < 1/1.5:
        raise Exception('image should be squarish')

    if im.size != (input_size, input_size):
        resize = ((int(input_size*aratio), input_size) if w > h
            else (input_size, int(input_size/aratio)))
        im = im.resize(resize).crop((0, 0, input_size, input_size))

    #im = equalize(im)
    return im

# TODO: support batch?
# TODO: try multiple variants (brightness, flip, zoom in/out, crop, etc?)

@app.post("/query")
async def post_query(image: UploadFile, limit: int = 5):
    try:
        im = await process_image(image)
    except Exception as e:
        return JSONResponse(status_code=400, content=dict(error=str(e)))
    t = ims_to_batch([ im ])

    with torch.no_grad():
        e = encoder(t.to(device=device)).cpu()[0]

    def go(idx, sim, anns, aid):
        m = ms_to_most_facing(msd[aid])
        fpath = path_to_thumb( m.f )
        imb64 = None
        if os.path.exists(fpath):
            with Image.open( fpath ) as im2:
                im2 = ImageOps.grayscale( im2 ) # they are RGBA
                imb64 = im_to_base64( im2, format='jpeg' )
        return dict_nonull(
            idx=idx,
            sim=sim,
            anns=anns,
            avatar=m.targetPlayer.avatar,
            image=dict_nonull(base64=imb64),
        )

    return dict(
            query=dict(base64=im_to_base64( im, format='jpeg' )),
            matches=[ go(**r) for r in query(embeds, e, limit=limit) ]
        )
