import subprocess
import time
from pyngrok import ngrok
import requests

def run_streamlit():
    # Streamlit uygulamasını başlat
    process = subprocess.Popen(["streamlit", "run", "app.py"])
    return process

def setup_ngrok():
    # ngrok auth token'ı ayarla
    ngrok.set_auth_token("2j4ohqLlsZ99rhNva7yIG2Uji0t_5fXU52P9j4XEHzFT9jHZV")
    
    # ngrok tünelini oluştur
    public_url = ngrok.connect(8501)
    print(f"Public URL: {public_url}")

def main():
    # Streamlit uygulamasını başlat
    streamlit_process = run_streamlit()

    # Streamlit'in başlaması için bekle
    print("Streamlit başlatılıyor...")
    time.sleep(10)

    # Streamlit'in çalışıp çalışmadığını kontrol et
    try:
        response = requests.get("http://localhost:8501")
        if response.status_code == 200:
            print("Streamlit başarıyla çalışıyor.")
            setup_ngrok()
        else:
            print(f"Streamlit çalışmıyor. Status code: {response.status_code}")
    except requests.exceptions.RequestException:
        print("Streamlit'e bağlanılamadı. Lütfen manuel olarak kontrol edin.")

    try:
        # Ana program döngüsü
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Uygulama kapatılıyor...")
    finally:
        # Temizlik işlemleri
        ngrok.kill()
        streamlit_process.terminate()

if __name__ == "__main__":
    main()
