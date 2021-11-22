ffmpeg -i cooker0930.mp4 -ss 00:38:00 -to 00:44:00 -vf crop=iw/2:ih/2:iw/4:0 nosteam/0930_%d.png
ffmpeg -i cooker0930.mp4 -ss 00:44:00 -vf crop=iw/2:ih/2:iw/4:0 steam/0930_%d.png
