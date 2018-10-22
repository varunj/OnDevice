for file in testvideo_imgs/*.jpg
do
	identify -format "%wx%h" $file
	echo ''
done
