wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lWdxgl10bHCbdYmYfnu7WgDfrY8nOcgT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lWdxgl10bHCbdYmYfnu7WgDfrY8nOcgT" -O best_ckpt.zip && rm -rf /tmp/cookies.txt

unzip best_ckpt.zip

mv best_ckpt/* .

rm -r best ckpt

rm best_ckpt.zip
