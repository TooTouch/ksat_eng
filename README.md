# ksat_eng

# Environments

```
pip install -r requirements.txt
```

# Data Scrapping

- **[category]**: 문제 유형. ex) '순서'
- **[EBSi ID]**: EBSi 아이디
- **[EBSi PASSWORD]**: EBSi 비밀번호

```bash
python scraping/main_scraping.py --problem_cat [category] --ebs_id [EBSi ID] --ebs_passwd [EBSi PASSWORD]
```

만약 중간에 에러가 생겼다면 아래 명령어

```bash
python scraping/main_scraping.py --resume --problem_cat [category] --ebs_id [EBSi ID] --ebs_passwd [EBSi PASSWORD]
```

그 외 argument에 대한 정보는 아래 명령어

```bash
python scrapping/main_scraping.py --help
```

# Download and Load Data

```python
from dataload import load_data

data = load_data(path='../data',
                 category='order',
                 download=True)
```
