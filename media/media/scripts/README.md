# Scripts
conda activate myenv
- gen_ptb_img.sh: Generate PTB-XL images from the PTB-XL dataset.

bash gen_ptb_img.sh --ptb-dir /home/fdias/data/CINC_CHALLENGE_2024/ptb-xl/1.0.3/records100/ --ptb-more more --ptb-more-img more_img --ptbxl-database-csv-dir /home/fdias/data/CINC_CHALLENGE_2024/ptb-xl/1.0.3/ptbxl_database.csv --ptbxl-scp-statements-dir /home/fdias/data/CINC_CHALLENGE_2024/ptb-xl/1.0.3/scp_statements.csv --store_text_bounding_box --store_config --bbox --augment -rot 40 -noise 40 --hw_text -n 4 --x_offset 30 --y_offset 20 --wrinkles -ca 45 --random_dc 0.5 --random_grid_present 0.8 --random_add_header 0.5 --random_resolution -r 200 --random_grid_color --fully_random -se 10

- list_png.sh: List all the png files in a directory.

- resize_images.py: Resize images in a directory and saves them in a new directory.
python resize_images.py --input /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/ALL/NOISE --output /home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/ALL/NOISE_224 --width 320 --height 320 --method resize