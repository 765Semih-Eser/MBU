#  RAG Tabanlı Üniversite Asistanı

## Kurulum

Gerekli komutları çalıştırmak için projenin kök dizinine geliniz ve CMD programını kök dizinde çalıştırınız.

Projenin düzgün çalışması için sanal ortamın oluşturulması tavsiye edilir. Sanal ortamı oluşturmak için şu komutu çalıştırınız:
```bash
python -m venv venv  
```

Oluşan sanal ortamı aktif etmek için şu komutu çalıştırınız:
```bash
venv\Scripts\activate
```

Projeyi çalıştırmadan önce şu komutu çalıştırınız: 
```bash
pip install -r .\requirements.txt 
```
Python 3.11, projenin çalışması için gereklidir

Projeyi başlatmak için şu komutu çalıştırınız: 
```bash
pyton -m streamlit run app.py  
```
Projeyi her başlattığınızda sanal ortamı tekrar aktifleştirmeniz gerekmektedir. Şu komutu girerek aktifleştirebilirsiniz:
```bash
venv\Scripts\activate
```
İşiniz bittiği zaman sanal ortamı kapatmak için şu komutu girebilirsiniz:
```bash
deactivate
```
