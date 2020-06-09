from os import walk
import os
import sys



# Change this path with yours.
# Also make sure that youtube-dl and ffmpeg installed.
# Previous versions of youtube-dl can be slow for downloading audio. Make sure you have downloaded the latest version from webpage.
# https://github.com/rg3/youtube-dl
# mypath = ""
# os.chdir(mypath)
def youtube_to_wav(link):
    os.system("\"youtube-dl\\youtube-dl.exe --extract-audio " + link+"\"")
    vidID = link.split("=")[1]
    print("VidID = " + vidID)
    f = []
    for (dirpath, dirnames, filenames) in walk("."):
        f.extend(filenames)
        break

    for i in range(0, len(f)):

        if ".opus" in f[i] and vidID in f[i]:
            vidName = f[i]
            print(vidName)
            cmdstr = "\"youtube-dl\\ffmpeg.exe -i \"" + vidName + "\" -f wav -flags bitexact \"" + vidName[:-5] + ".wav" + "\"\""
            print(cmdstr)
            os.system(cmdstr)
            os.remove(vidName)  # Will remove original opus file. Comment it if you want to keep that file.
            print(vidName[:len(vidName)-5]+".wav")
            return vidName[:len(vidName)-5]+".wav"
