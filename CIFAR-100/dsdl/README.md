# Data Set Description Language(DSDL) for CIFAR-100 dataset

## prepare the dataset
To make sure the DSDL dataset for Image Classification run successfully, the tools/prepare.py should be executed. 
For this dataset, the following step will be selected to execute:
- decompress
- prepare dataset and generate DSDL annotation

There are four usage scenarios:
```
### decompress, convert
python tools/prepare.py <path_to_the_compressed_dataset_folder>

### decompress, copy and convert
python tools/prepare.py -c <path_to_the_compressed_dataset_folder>

### (already decompressed) copy and convert
python tools/prepare.py -d -c <path_to_the_decompressed_dataset_folder>

### (already decompressed) convert, directly overwrite
python tools/prepare.py -d <path_to_the_decompressed_dataset_folder>
```

For more messages, see [Dataset Prepare Section](https://opendatalab.github.io/dsdl-docs/tutorials/dataset_download/) in DSDL DOC, or use the help option:

```
python tools/prepare.py --help
```
    
## Data Structure
Please make sure the folder structure of prepared dataset is organized as followed:

```
<dataset_root>
├── file.txt~
├── meta
├── test
└── train

```

The folder structure of dsdl annotation for Image Classification is organized as followed:

```
<dsdl_root>
├── defs
│   ├── class-dom.yaml
│   └── template.yaml
├── tools
│   └── prepare.py
├── set-train
│   ├── train.yaml
│   └── train_samples.json
├── set-test
│   ├── test.yaml
│   └── test_samples.json
├── README.md
└── config.py

```

## config.py
You can load your dataset from local or oss.
From local:

```
local = dict(
    type="LocalFileReader",
    working_dir="the root path of the prepared dataset",
)
```

Please change the 'working_dir' to the path of your prepared dataset where media data can be found,
for example: "<root>/dataset_name/prepared".

From oss:

```
ali_oss = dict(
    type="AliOSSFileReader",
    access_key_secret="your secret key of aliyun oss",
    endpoint="your endpoint of aliyun oss",
    access_key_id="your access key of aliyun oss",
    bucket_name="your bucket name of aliyun oss",
    working_dir="the prefix of the prepared dataset within the bucket")
```

Please change the 'access_key_secret', 'endpoint', 'access_key_id', 'bucket_name' and 'working_dir',
e.g. if the full path of your prepared dataset is "oss://bucket_name/dataset_name/prepared", then the working_dir should be "dataset_name/prepared".

## Related source:
1. Get more information about DSDL: [dsdl-docs](https://opendatalab.github.io/dsdl-docs/)
2. DSDL-SDK official repo: [dsdl-sdk](https://github.com/opendatalab/dsdl-sdk/)
3. Get more dataset: [OpenDataLab](https://opendatalab.com/)
