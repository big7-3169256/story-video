"""Turns story.json into final.mp4 (1080x1920) using only free tools.

story.json shape:
{
  "title": "...", "character": "...", "style": "...", "voice": "en-US-ChristopherNeural" (optional),
  "scenes": [{"text": "narration", "image_prompt": "what is visible"}]
}
"""
import asyncio
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request

import edge_tts

W, H, FPS = 1080, 1920, 30
PAD = 0.3  # seconds of silence after each scene
WORK = "work"
os.makedirs(WORK, exist_ok=True)

story = json.load(open("story.json", encoding="utf-8"))
VOICE = story.get("voice") or "en-US-ChristopherNeural"
STYLE = story.get("style") or "flat 2D cartoon illustration, bold outlines, vibrant colors"
CHARACTER = story.get("character") or ""


def run(cmd):
    print(" ".join(cmd)[:200])
    subprocess.run(cmd, check=True)


def fetch_image(prompt, seed, path):
    full = f"{STYLE}, {CHARACTER}, {prompt}, vertical composition, no text, no watermark"
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(full)
        + f"?width={W}&height={H}&seed={seed}&nologo=true&model=flux"
    )
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=180).read()
            if len(data) > 10000:
                with open(path, "wb") as f:
                    f.write(data)
                return
        except Exception as e:  # noqa: BLE001
            print("image retry:", e)
        time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"Image generation failed for: {prompt[:60]}")


async def tts(text, mp3_path):
    comm = edge_tts.Communicate(text, VOICE, rate="+8%", boundary="WordBoundary")
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                words.append((start, end, chunk["text"]))
    return words


def audio_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path]
    )
    return float(out.strip())


def ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(all_words, path):
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,DejaVu Sans,88,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,7,2,2,60,60,420,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    ) % (W, H)
    lines = []
    chunk_size = 3
    for i in range(0, len(all_words), chunk_size):
        group = all_words[i:i + chunk_size]
        start = group[0][0]
        end = group[-1][1]
        # keep the caption up until the next chunk starts so there is no flicker
        if i + chunk_size < len(all_words):
            end = max(end, all_words[i + chunk_size][0])
        text = " ".join(w[2] for w in group).upper().replace("\n", " ")
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def make_clip(img, mp3, dur, idx, out):
    frames = int(dur * FPS) + 1
    if idx % 2 == 0:
        z = f"1+0.22*on/{frames}"        # slow zoom in
    else:
        z = f"1.22-0.22*on/{frames}"     # slow zoom out
    vf = (
        f"scale={W * 2}:{H * 2},"
        f"zoompan=z='{z}':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS},"
        "format=yuv420p"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", img, "-i", mp3,
        "-vf", vf, "-af", f"apad=pad_dur={PAD}", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", out,
    ])


async def main():
    clips, all_words, offset = [], [], 0.0
    for i, scene in enumerate(story["scenes"]):
        img = f"{WORK}/scene{i}.png"
        mp3 = f"{WORK}/scene{i}.mp3"
        clip = f"{WORK}/clip{i}.mp4"

        words = await tts(scene["text"], mp3)
        dur = audio_duration(mp3) + PAD
        fetch_image(scene["image_prompt"], seed=1000 + i, path=img)
        make_clip(img, mp3, dur, i, clip)

        all_words += [(s + offset, e + offset, t) for s, e, t in words]
        offset += dur
        clips.append(clip)

    with open(f"{WORK}/list.txt", "w") as f:
        for c in clips:
            f.write(f"file '{os.path.basename(c)}'\n")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", f"{WORK}/list.txt", "-c", "copy", f"{WORK}/joined.mp4"])

    build_ass(all_words, f"{WORK}/captions.ass")

    # optional background music: drop a royalty-free file at assets/music.mp3
    if os.path.exists("assets/music.mp3"):
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{WORK}/joined.mp4",
             "-stream_loop", "-1", "-i", "assets/music.mp3",
             "-filter_complex",
             f"[0:v]ass={WORK}/captions.ass[v];[1:a]volume=0.07[m];[0:a][m]amix=inputs=2:duration=first[a]",
             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "aac", "-shortest", "final.mp4"])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{WORK}/joined.mp4",
             "-vf", f"ass={WORK}/captions.ass", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "copy", "final.mp4"])
    print("done: final.mp4", round(offset, 1), "seconds")


asyncio.run(main())
