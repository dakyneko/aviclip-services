#!/usr/bin/env python
# coding: utf-8

from discord import *
import os, io, logging, requests
from PIL import Image, UnidentifiedImageError
from fastcore.basics import AttrDict
from collections import Counter
from functools import partial
from base64 import b64decode
from io import BytesIO

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


client = Client()

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
        await process_message(m)
    except:
        await exploded(m)
        exception('boom')

async def process_message(m):
    if not any(( t in m.content for t in ('sauce', 'source') )):
        return await confused(m, 'message without request')

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
            with m.channel.typing():
                return await process_attachment(m, a)

    return await confused(m, f'no attachment found anywhere')


def bunchize(x):
    if isinstance(x, list):
        return [ bunchize(y) for y in x ]
    if isinstance(x, dict):
        return AttrDict({ k:bunchize(v) for k,v in x.items() })
    else:
        return x


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
                params=dict(limit=5),
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
    wtotal = sum(( im.width for im in ims ))
    hmax = max(( im.height for im in ims  ))
    montage = Image.new('RGB', (wtotal, hmax))

    # horizontally
    x = 0
    for im in ims:
        montage.paste(im, (x, 0))
        x += im.width

    bio = BytesIO()
    montage.save(bio, format='jpeg', quality=85)
    bio.seek(0) # prepare for reading

    def go(idx, sim, avatar, anns, **kwargs):
        s = f'- {sim*100:.1f}% '
        s += anns.name if 'name' in anns else '(unknown)'
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
            file=File(bio, filename='results.jpg'))

    info('results: '+ ', '.join(( f'#{x.idx} {x.sim*100:.1f}% "{x.avatar.id}"' for x in xs )))

    # TODO: revisit implementation
    #return await process_matches(m, xs)


def anns_counter(xs, attr):
    return dict(Counter(( x.anns.get(attr, None) for x in xs )).most_common())

def guess_from_attr(cnt_by_attr, attr_name, xs, best_score):
    if list(cnt_by_attr.keys()) == [None]:
        cnt = cnt_by_attr[None]
        if cnt == len(xs):
            s = f"- All {cnt}"
        else:
            s = f"- The {cnt} of the {len(xs)}"
        s += f" best matches have unknown {attr_name}."
    elif len(cnt_by_attr) == 1:
        q = 'good' if best_score >= good_score else 'approximate'
        v, cnt = next(iter(cnt_by_attr.items()))
        s = f"- I found a {q} match with {attr_name} '{v}'"
        if cnt_by_attr[v] > 1:
            s += f" (found {cnt} times)"
    elif len(cnt_by_attr) == 2 and None in cnt_by_attr:
        cntNone = cnt_by_attr[None]
        del cnt_by_attr[None]
        v, cnt = next(iter(cnt_by_attr.items()))
        s = f"- I found {cntNone} matches for which we don't know their {attr_name} and {cnt} others is {v}"
    else:
        s = f"- The best matches have different {attr_name}s, it's unconclusive."
    return [ s ]


async def process_matches(m, xs):
    # summarize the matches
    xs_min  = [ x for x in xs if x.sim > min_score ]
    has_min_match = len(xs_min) > 0

    ss = []

    # group close to 1st best match
    if has_min_match:
        ss.append( "\N{WHITE HEAVY CHECK MARK} I took your image and run a query against my database. Here are the results:" )
        best_score = xs_min[0].sim
        close_score = best_score - 0.05
        xs_bests = [ x for x in xs if x.sim >= close_score ]

        by_names = anns_counter(xs_bests, 'name')
        by_creator = anns_counter(xs_bests, 'creator')
        by_category = anns_counter(xs_bests, 'category')

        # TODO: for name mention the creator if present
        guess = partial(guess_from_attr, xs=xs_bests, best_score=best_score)
        ss.extend(guess(by_names, 'name'))
        ss.extend(guess(by_creator, 'creator'))
        ss.extend(guess(by_category, 'category'))
    else:
        ss.append( "\N{CROSS MARK} Unfortunately, I run a query but nothing came up similar enough. If you have more pictures, you can try them." )

    #s = "I searched but found no conclusive match"
    await m.reply( ss[0] + '\n' + '\n'.join(ss[1:]) )


# let's go!
client.run(os.environ['DISCORD_BOT_TOKEN'])
