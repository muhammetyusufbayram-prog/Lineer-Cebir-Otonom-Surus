import cv2
from ultralytics import YOLO

# Ödev projemiz için hazır eğitilmiş yolov8 segmentasyon modelini yüklüyoruz
model = YOLO('yolov8n-seg.pt')

# Analiz edeceğimiz orijinal video dosyasının adı
video_dosyasi = 'trafik.mp4'
cap = cv2.VideoCapture(video_dosyasi)

# Giriş videosunun boyutlarını ve fps değerini çıktı videosu için alıyoruz
genislik = int(cap.get(3))
yukseklik = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Windows ortamında en rahat açılan video formatı ayarı (mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_segmentasyon.mp4', fourcc, fps, (genislik, yukseklik))

print("Görüntü işleme başladı, lütfen işlem bitene kadar bekleyin...")

# Videoyu kare kare okumak için döngü başlatıyoruz
while cap.isOpened():
    ret, kare = cap.read()
    if not ret:
        break  # Video bittiğinde döngüden çık

    # Yapay zeka modeliyle nesneleri tespit edip maskeleri kare üzerine çizdiriyoruz
    sonuclar = model(kare, verbose=False)
    islenmis_kare = sonuclar[0].plot()

    # Karar Destek Sistemi için başlangıçtaki güvenli durumumuz
    durum = "GUVENLI SURUS"
    renk = (0, 255, 0)  # Yeşil renk (BGR formatında)
    
    # Eğer ekranda herhangi bir nesne algılandıysa alan hesaplarına geçiyoruz
    if sonuclar[0].boxes is not None and len(sonuclar[0].boxes) > 0:
        for kutu in sonuclar[0].boxes:
            # Algılanan nesnenin ekran üzerindeki köşe koordinatları
            koordinat = kutu.xyxy[0].tolist()
            x1, y1, x2, y2 = koordinat[0], koordinat[1], koordinat[2], koordinat[3]
            
            # Nesnenin pikselsel alanının tüm ekrana olan yüzdesel oranını buluyoruz
            nesne_alani = (x2 - x1) * (y2 - y1)
            toplam_alan = genislik * yukseklik
            oran = (nesne_alani / toplam_alan) * 100
            
            # Nesnenin sınıf numarasını alıyoruz (0: insan, 2: araba, 7: kamyon)
            sinif_id = int(kutu.cls[0].item())
            
            if sinif_id in [0, 2, 7]:
                # Eğer araç çok yakınsa ve ekranın yüzde 12'sinden fazlasını kaplıyorsa
                if oran > 12.0:
                    durum = "ACIL FREN YAP!"
                    renk = (0, 0, 255)  # Kırmızı renk
                    break  # En kritik durumu bulduğumuz için diğer nesnelere bakmadan çıkıyoruz
                # Eğer araç orta mesafedeyse ve takip mesafesi riskliyse
                elif 5.0 < oran <= 12.0:
                    durum = "HIZI AZALT / TAKIP MESAFESI"
                    renk = (0, 255, 255)  # Sarı renk

    # Ürettiğimiz otonom sürüş kararını videonun sol üst köşesine yazdırıyoruz
    cv2.putText(islenmis_kare, f"KDS KARARI: {durum}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 3, cv2.LINE_AA)
    
    # Üzerine yazı yazılmış kareyi çıktı videomuza ekliyoruz
    out.write(islenmis_kare)

# İşlem bittiği için arka plandaki tüm kaynakları kapatıyoruz
cap.release()
out.release()
print("İşlem tamamlandı! 'output_segmentasyon.mp4' dosyası başarıyla oluşturuldu.")