#!/usr/bin/env python
# coding: utf-8

from discord import *
from discord.ui import *
import os, io, gzip, logging, requests, yaml, itertools, re, json, random
from PIL import Image, UnidentifiedImageError
from fastcore.basics import AttrDict
from collections import Counter
from functools import partial
from base64 import b64decode
from io import BytesIO
from time import time
from collections import defaultdict

# Q: could replace requests by grequests for async?


api_url = 'http://localhost:8000'
good_score = 0.9
min_score = 0.85


def setup_logs():
    rl = logging.getLogger()
    rl.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)5s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.FileHandler('discord_bot.log')
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    rl.addHandler(ch)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    rl.addHandler(ch)

    l = logging.getLogger('main')
    l.setLevel(logging.DEBUG)

    return l

l = setup_logs()
debug = l.debug; info = l.info; warn = l.warning; error = l.error; exception = l.exception


client = Client(intents=Intents.default())

@client.event
async def on_ready():
    info(f'Logged on as {client.user}!')


def msg_to_str(m):
    s = f'{m.guild} {m.author}'

    c = m.channel
    if type(c) == TextChannel:
        s += f' | in {c.category}/{c.name}'
    elif type(c) == DMChannel:
        s += f' | DM {c.recipient}'
    else:
        s += f' | unknown channel ({type(c)}): {c}'

    mentionned = client.user in m.mentions
    if len(m.mentions) > 0:
        flag = "[!]" if mentionned else ""
        s += f' | mentions{flag}: '
        s += ', '.join(( x.name for x in m.mentions ))

    if len(m.attachments) > 0:
        s += f' | {len(m.attachments)} attachments: '
        s += ', '.join(( f'{a.content_type} {a.filename}[{a.url}]' for a in m.attachments ))

    if m.reference:
        r = m.reference
        s += f' | ref: {r.guild_id}/{r.channel_id}/{r.message_id}'
        if r.cached_message:
            s += f' \nfrom cache: {msg_to_str(r.cached_message)}'

    if m.content: # we need intent message to see other's messages
        s += f'\n{m.content}'

    return s


async def confused(m, reason=None):
    await m.add_reaction('\N{WHITE QUESTION MARK ORNAMENT}')
    info(f'unknown request {reason}: {m}')

async def failed(m, reason=None):
    await m.add_reaction('\N{CROSS MARK}')
    return False

async def exploded(m):
    await m.add_reaction('\N{SKULL AND CROSSBONES}')


@client.event
async def on_message(m):
    if m.author == client.user: return

    info(f'on_message {msg_to_str(m)}')

    # TODO: save other's image requests+replies = we learn!
    if client.user not in m.mentions: return # Q: reply to private msg?

    # bot is addressed
    try:
        if any(( t in m.content for t in ('sauce', 'source') )):
            await process_message_sauce(m)
        elif any(( t in m.content for t in ('next', 'group') )):
            await process_message_annotate(m)
        elif any(( t in m.content for t in ('search',) )):
            await process_message_search(m)
        else:
            return await confused(m, 'message without request')
    except:
        exception('boom')
        await exploded(m)


def bunchize(x):
    if isinstance(x, list):
        return [ bunchize(y) for y in x ]
    if isinstance(x, dict):
        return AttrDict({ k:bunchize(v) for k,v in x.items() })
    else:
        return x

def chunker(iterable, n):
    if type(iterable) == list:
        iterable = iter(iterable)
    while True:
        x = next(iterable, None)
        if x != None:
            yield [x] + list(itertools.islice(iterable, n-1))
        else:
            break

def autocrop(im):
    return im.crop( im.getbbox() )


with gzip.open('annotations_rs-vs3_gs_todo.yaml.gz', 'rt') as fd:
    gs = list(bunchize(yaml.load(fd, Loader=yaml.Loader)))
info(f'loaded {len(gs)} groups for annotations')
with open('annotations_rs-vs3_gs_logs.jsonl', 'rt') as fd:
    lanns = {}
    user_scores = defaultdict(lambda: AttrDict(user=None, score=0))
    for l in fd:
        j = bunchize(json.loads(l))
        lanns[j.i] = j # this replace any prior annotation

        user_score = user_scores[j.user.id]
        user_score.user = j.user
        user_score.score += 1
info(f'loaded {len(lanns)} logs for annotations')

def write_lann(lann):
    with open('annotations_rs-vs3_gs_logs.jsonl', 'at') as fd:
        fd.write(json.dumps(lann) + '\n')


async def process_message_annotate(m):
    try:
        (idx,) = re.findall(r'group #?(\d+)', m.content)
        g = gs[int(idx)]
    except ValueError:
        g = next(( g for g in gs if g.i not in lanns ))

    info(f'prepare annotation {g.i}')

    s = [ f'**Group #{g.i}** with {len(g.idents)} entries\n',
            f'**Common words**:', ]
    s.extend(( f'- {w}: {cnt}' for w, cnt in g.names.items() ))
    s.append('')
    if len(g.hints) > 0:
        s.append(f'**Hints**:')
        s.extend(( f'- {h}' for h in g.hints ))
    else:
        s.append('without any hint')

    already_sauced = g.i in lanns
    if already_sauced:
        s.append('\n\N{WARNING SIGN} **Note**: This group was already sauced')

    def go(f):
        im = autocrop(Image.open(f'../dataset_ripperstore/{f}'))
        bg = Image.new('RGB', im.size, 3*(255,))
        return Image.composite(im, bg, im)
    ims = [ go(f) for f in g.imgs_vdedup ]
    bio = ims_to_montage(ims)

    await m.reply('\n'.join(s),
            file=File(bio, filename='results.jpg'),
            view=MyView(g),
        )

# TODO: create this view dynamically above, simpler + can change button label if already sauced!
class MyView(View):
    def __init__(self, g):
        super().__init__()
        self.g = g

    @button(label='sauce it', style=ButtonStyle.blurple)
    async def receive(self, interaction: Interaction, button: Button):
        await interaction.response.send_modal(MyModal(g=self.g))

modal_fields = 'name creator url comment'.split()

class MyModal(Modal, title='Saucing an avatar, yay'):
    name = TextInput(
        label='Name',
        required=False,
    )
    creator = TextInput(
        label='Creator',
        required=False,
    )
    url = TextInput(
        label='URL to buy',
        placeholder='https://something.gumroad.com',
        required=False,
    )
    comment = TextInput(
        label='Comment',
        placeholder='eg: TDA, cannot buy, discord nitro only, etc',
        style=TextStyle.long,
        required=False,
    )

    def __init__(self, g):
        # restore values we have if they exists so they can be edited easily
        if g.i in lanns:
            lann = lanns[g.i]
            for k in modal_fields:
                if k in lann:
                    f = getattr(self, k)
                    f.default = lann[k]

        super().__init__()
        self.g = g

    async def on_submit(self, interaction: Interaction):
        data = AttrDict({ k: getattr(self, k).value for k in modal_fields })
        user = interaction.user
        g = self.g
        info(f'Modal data from {user} for group {g.i}: {data}')

        data.i = g.i
        data.user = AttrDict(id=user.id, name=user.name)
        data.time = int(time())

        lanns[g.i] = data
        write_lann(data)

        user_score = user_scores[user.id]
        user_score.user = data.user
        user_score.score += 1

        # TODO: diversify messages, look nicer
        # could send messages prop to number of points
        s1 = random.choice([
            "Thankies! \N{TWO HEARTS}",
            "is so kind! \N{BLACK HEART SUIT}",
            "\N{ORANGE HEART} Thank you for your help!",
            "Your contributions are very appreciated!",
            "That's awesome, thank you \N{SMILING CAT FACE WITH HEART-SHAPED EYES}",
            "*headpats*",
            "*cuddles* \N{SMILING CAT FACE WITH HEART-SHAPED EYES}",
            "You deserve a medal \N{FIRST PLACE MEDAL}",
            "So sweet! \N{SMILING FACE WITH SMILING EYES AND THREE HEARTS}",
            "You're a sweetie \N{SPARKLING HEART}",
            "Yes you're cute",
            "Very appreciated \N{SPARKLES}"
        ])
        s2 = random.choice([
            "with {score} lovely contributions!",
            "with a total of {score} contributions!",
            "{score} love points!",
            "{score} million headpats, I mean points!",
            "{score} thousand cuddles, I mean points!",
            "You reached {score} points!",
            "Look, now {score} points, congrats!",
            "Unbeatable {score} contributions!",
        ])
        await interaction.response.send_message(
                f'<@{user.id}> ' + s1 + ' ' + s2.format(score=user_score.score),
                ephemeral=False, # keep public?
                )

    async def on_error(self, interaction: Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Something went wrong.', ephemeral=True)

        # Make sure we know what the error actually is
        exception(error)


async def process_message_search(m):
    try:
        (text,) = re.findall(r'search (.+)', m.content)
    except ValueError:
        return await confused(m, f'search missing text query?')

    if len(text) < 3:
        return await confused(m, f'search text query too short')

    text = text.lower()
    gs2 = [ g
            for g in gs
            if all(( any(( w in n for n in g.names  )) for w in text.split() )) ]
    total = len(gs2)
    gs2 = gs2[:5]

    s = [ f'Found {total} groups matching \'{text}\':\n']

    if len(gs2) == 0:
        return # done

    s.append(f'**Groups**:')
    def go(i, idents, names, **kwargs):
        s = [f'- **Group #{i}** with {len(idents)} samples']
        s.append('  common words: ' + ', '.join(( f'{w} ({cnt}x)' for w, cnt in names.items() )))
        return '\n'.join(s) + '\n'
    s.extend(( go(**g) for g in gs2  ))

    if len(gs2) != total:
        s.append("\N{WARNING SIGN} **Note**: More groups matched but aren't shown here")

    await m.reply('\n'.join(s))


async def process_message_sauce(m):
    ms = [ m ]
    if m.reference:
        r = m.reference
        if not r.cached_message:
            info(f'ignoring reference message (not in cache): {r}')
        else:
            ms.append( r.cached_message )

    for m in ms:
        cnt = len(m.attachments)
        if cnt > 1:
            return await confused(m, f'too many attachments')
        elif cnt == 1:
            a = m.attachments[0]
            if not a.content_type or not a.content_type.startswith('image/'):
                return await confused(m, f'attachment of unknown type: {a.content_type} for {a.filename}')
            async with m.channel.typing():
                return await process_attachment(m, a)

    return await confused(m, f'no attachment found anywhere')


def ims_to_montage(ims, cols=6, format='jpeg', quality=85):
    wtotal = max(( sum(( im.width for im in ims_ )) for ims_ in chunker(ims, cols) ))
    htotal = sum(( max(( im.height for im in ims_ )) for ims_ in chunker(ims, cols)  ))
    montage = Image.new('RGB', (wtotal, htotal), 3*(255,))

    # in grid
    y = 0
    for ims_ in chunker(ims, cols):
        x = 0
        for im in ims_:
            montage.paste(im, (x, y))
            x += im.width
        y += max(( im.height for im in ims_ ))

    bio = BytesIO()
    montage.save(bio, format=format, quality=quality)
    bio.seek(0) # prepare for reading
    return bio

async def process_attachment(m, a):
    info(f'processing attachment mine={a.content_type} size={a.size/1024:.2f}KB at {a.url}')
    # load the image bytes
    try:
        raw = await a.read()
    except:
        exception('failure to download attachment: {a.url}')
        return await failed(m)

    # test reading image with PIL
    try:
        im = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as e:
        exception('image format unknown')
        return await failed(m)
    except Exception as e:
        exception('unknown error while opening image')
        return await failed(m)

    info(f'image open: mode={im.mode} size={im.size}')
    im.close()

    # query API
    try:
        r = requests.post(f'{api_url}/query',
                params=dict(limit=6),
                files=dict(image=io.BytesIO(raw)),
            )
        j = bunchize(r.json())
    except:
        exception(f'an exception occured during the api request')
        return await failed(m)

    if r.status_code == 400:
        warn(f'API rejected request {r.status_code}: {j.error}')
        return await m.reply( f"\N{CROSS MARK} Sorry I couldn't run the query: {j.error}" )
    elif r.status_code != 200:
        error(f"API error {r.status_code}: {j.get('error', None)}")
        return await failed(m)

    xs = j.matches
    if len(xs) == 0:
        error(f'API returned 0 match')
        return await failed(m)

    # compose together all images for easy display
    # TODO: try pyplt to make montage table with columns, +legends numbers?
    ims = [ Image.open(BytesIO(b64decode( x.image.base64 )))
            for x in xs if 'base64' in x.image ]
    bio = ims_to_montage(ims)

    def go(idx, sim, avatar, anns, **kwargs):
        s = f'- {sim*100:.1f}% '
        s += anns.get('name', '(unknown)')
        if 'creator' in anns:
            s += f' by {anns.creator}'
        if 'category' in anns:
            s += f' in the {anns.category} category'
        # TODO: do we want to expose those (private) info?
        s += f', upload name: "{avatar.name}"'
        s += f' #{idx}'
        return s

    await m.reply(
            'Closest similar matches:\n' + '\n'.join([go(**x) for x in xs]),
            file=File(bio, filename='results.jpg'),
            )

    info('results: '+ ', '.join(( f'#{x.idx} {x.sim*100:.1f}% "{x.avatar.id}"' for x in xs )))


# let's go!
client.run(os.environ['DISCORD_BOT_TOKEN'])
