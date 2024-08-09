# Check if the source folder is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 source_folder"
    exit 1
fi

cd "$1"
git clone https://github.com/physionetchallenges/python-example-2024.git
git clone https://github.com/physionetchallenges/evaluation-2024.git

cd python-example-2024
mv add_image_filenames.py helper_code.py prepare_ptbxl_data.py remove_hidden_data.py run_model.py train_model.py ../
mv model/digitization_model.sav ../.
cd ..
cd evaluation-2024
mv evaluate_model.py ../
cd ..

mv digitization_model.sav ../model/
rm -rf python-example-2024 evaluation-2024