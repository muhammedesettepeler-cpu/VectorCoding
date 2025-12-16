# Data Directory (`data_dir`)

Bu klasör, projenin veri setlerini ve veri konfigürasyonlarını barındırır.

## Yapı

*   **`master_config.yaml`**: Tüm senaryolar için veri seti ayarlarını içeren ana konfigürasyon dosyası. Hangi veri setinin aktif olduğunu ve her birinin detaylarını buradan yönetebilirsiniz.
*   **Alt Klasörler (Örn: `sentiment`, `news_articles`)**: Her bir senaryo veya veri seti için ayrılmış klasörlerdir. Bu klasörler, ilgili veri dosyalarını (genellikle `.parquet` formatında) içerir.

## Notlar

*   Büyük veri dosyaları (örneğin `.parquet`, `.csv`) GitHub deposuna **yüklenmez**.
*   Sadece konfigürasyon dosyaları ve bu README dosyası versiyon kontrolündedir.
*   Veri dosyalarını yerel ortamınıza manuel olarak eklemeniz veya bir veri kaynağından indirmeniz gerekebilir.
