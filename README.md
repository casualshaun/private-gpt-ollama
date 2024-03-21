# private-gpt-ollama
A private GPT using ollama

# Installation

## 1. Set up Virtual Environment

### Verify you have conda installed
If you don't have conda installed go to the [Anaconda Distro Page](https://docs.anaconda.com/free/distro-or-miniconda/)

If you already have conda ensure you have the latest version
```
conda update conda
```

### Create Virtual Enviroment
```
conda create -n NAME_OF_PRIVATE_GPT
```

### Activate the Virtual Enviroment
```
conda activate NAME_OF_PRIVATE_GPT
```

## 2. Install the Requirements
```
pip install -r requirements.txt
```

## 3. Pull the Models

### You will need ollama running already. You can get [Ollama here](https://ollama.com/)

You can use any model you want, I just went with mistral. You can look at the many models [here](https://ollama.com/library)
```
ollama pull mistral
```

## 4. Create the Source Directory
```
mkdir source_docs
```

## 5. Ingest the Files in the Source Directory
Add any files with the valid extentions found in the LOADER_MAPPING into the source_docs folder. 
Then run the following:
```
python ingest.py
```

## 6. Start the GPT!
```
python private_gpt.py
```

