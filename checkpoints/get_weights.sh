wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1227yL-vIUhWYH8Idl2hpdiFloxZJUNq6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1227yL-vIUhWYH8Idl2hpdiFloxZJUNq6" -O best_ckpt.zip && rm -rf /tmp/cookies.txt

unzip best_ckpt.zip

rm best_ckpt.zip
