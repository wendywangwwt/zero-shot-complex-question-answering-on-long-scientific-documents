## PDF-to-Text Extraction
0. Software Installation
To install Tika:
```
pip install tika
```

To install Gobid ([link](https://grobid.readthedocs.io/en/latest/Install-Grobid/)):
```
wget https://github.com/kermitt2/grobid/archive/0.8.1.zip
unzip 0.8.1.zip
```

And to start Grobid service ([link](https://grobid.readthedocs.io/en/latest/Grobid-service/)):
```
cd grobid
./gradlew run
```

2. Tika (see folder "parser")
Example command:
```
python extract_pdf_text_tika.py --dir_data /scratch/papers
```

2. Label Studio
Do manual annotation for sections in label studio.


3. Grobid - used for problematic files that cannot be handled properly by Tika (usually you do not realize Tika provides a corrupted or bad extraction until you see it in label studio, hence step 3)

Get the grobid output first. It is easier to use the [python client](https://github.com/kermitt2/grobid_client_python) for a batch of pdf files:
```
python -m pip install grobid-client-python

grobid_client --config grobid_client_python/config.json --input  ./grobid_input/ --output ./grobid_output/ processFulltextDocument
```

Then use teitocsv to extract the text from the tei.xml output files.
```
!git clone https://github.com/komax/teitocsv
cd ./teitocsv/teitocsv

python main.py ../../data_grobid_output/ ../../data_grobid_output/data_grobid.csv
```

* This whole pdf-to-text process could possibly be replaced by **a large language model**, assuming its output is accurate despite of its generative nature (which I'm still a bit doubtful of).
