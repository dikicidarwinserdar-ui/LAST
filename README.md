# CDP Verify Camera

FastAPI + OpenCV + canlı kamera tabanlı CDP doğrulama test servisi.

## Ana akış

Bu versiyonda manuel fotoğraf çekimi yoktur. Kullanıcı sadece aşama seçer:

1. Referans Çek
2. Orijinal Çek
3. Kopya Çek
4. Test Et

Her aşamada kamera açılır. Sistem canlı görüntüde kalite kontrolü yapar:

- Netlik / focus
- Işık / brightness
- Kontrast
- Hareket / motion stability

Görüntü kalite eşiğini geçip kısa süre stabil kaldığında fotoğraf otomatik çekilir ve backend'e kaydedilir.

## Kayıt klasörleri

Referans çekimleri:

```text
data/references/
```

Diğer çekimler:

```text
data/captures/original/
data/captures/copy/
data/captures/test/
```

## Önemli not

Render free runtime'da deploy sonrası runtime içinde kaydedilen dosyalar restart sonrası kalıcı olmayabilir. İlk test için yeterlidir. Gerçek production sisteminde referans ve çekimler S3, Supabase Storage, Cloudinary veya benzeri kalıcı object storage'a taşınmalıdır.

## Local çalıştırma

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Aç:

```text
http://127.0.0.1:8000
```

## Render ayarları

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Environment:

```text
PYTHON_VERSION=3.11.11
```

## API

```text
GET  /health
GET  /api/refs
GET  /api/captures
POST /api/reload-refs
POST /api/capture/reference
POST /api/capture/original
POST /api/capture/copy
POST /api/capture/test
POST /api/verify
```

## İlk kullanım sırası

1. Siteyi aç.
2. `Referans Çek` ile en az 1 referans kaydet.
3. `Orijinal Çek` ile gerçek baskıyı test et.
4. `Kopya Çek` ile kopya baskıyı test et.
5. `Test Et` ile doğrulama akışını dene.

## Kamera kalite eşikleri

Frontend kalite eşikleri `static/camera.js` içinde tanımlıdır:

```js
const QUALITY = {
  minFocus: 42,
  minBrightness: 45,
  maxBrightness: 215,
  minContrast: 22,
  maxMotion: 8.5,
  stableMs: 650
};
```

Bu değerler kullanıcıyı fazla zorlamadan bulanık ve hareketli çekimleri engellemek için başlangıç ayarıdır.
