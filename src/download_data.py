"""Download the original public FORCE archive; never overwrite an existing CSV."""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = 'https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/train.zip'

def main():
    target = Path(__file__).resolve().parent.parent / 'data/raw/train.csv'
    if target.exists():
        raise SystemExit(f'O ficheiro já existe: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        archive = Path(temporary) / 'train.zip'
        print('A descarregar o dataset público FORCE (~92 MB). Consulte os termos da fonte no README.')
        urlretrieve(URL, archive)
        with ZipFile(archive) as zipped, zipped.open('train.csv') as source:
            import shutil
            staging = target.with_suffix('.csv.part')
            try:
                with staging.open('wb') as output:
                    shutil.copyfileobj(source, output)
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
    print(f'Dataset guardado em {target}')

if __name__ == '__main__':
    main()
