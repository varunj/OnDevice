for file in testvideo/*.mp4
do
	NAME=`basename $file .mp4`
	ffmpeg -i $file testvideo_imgs/$NAME"_"%03d.jpg;
done
