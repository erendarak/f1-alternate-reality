# F1 ALTERNATE REALITY — LIVING CHAMPIONSHIP SIMULATOR

> **Master Product Requirements Document (PRD) + Technical Design Document (TDD) + AI Research Plan + Claude Code Execution Guide**
>
> **Sürüm:** 1.0 (10 Ekim 2026)  
> **Dil:** Türkçe açıklamalar; kaynak kodu, API, test adları ve teknik isimler İngilizce.  
> **Hedef:** Tek geliştirici, haftada ortalama 6–10 saat, yerel Windows geliştirme ortamı.  
> **İlk tarihsel başlangıç:** 2024 Italian Grand Prix (Monza), 2024 sezonundaki gerçek kadrolar ve kurallar.  
> **Nihai ürün:** Sürekli ilerleyen, durdurulabilen/kaydedilebilen/yeniden açılabilen, dallanan alternatif tarihler oluşturabilen, karakter özellikleriyle koşullanan ve zamanla öğrenen F1 evreni.

---

## 0. CLAUDE CODE'A TALİMAT — ÖNCE BUNU OKU

Sen (`Claude Code`) bu belgenin uygulayıcı yazılım mühendisi ve teknik proje ortağısın. Amaç tüm projeyi tek oturumda üretmek değil, **belgedeki kilometre taşlarını sırasıyla, test ederek ve gerçekten çalıştırarak geliştirmek**. Hiçbir aşamada sahte çalışan ekranlar veya doğrulanmamış özellikleri tamamlanmış gösterme.

**Zorunlu çalışma kuralları**

1. Önce proje klasörünü incele. Klasör boşsa temel repoyu kur; doluysa mevcut kodu koruyup durum analizi yap. Var olan dosyaları sebepsiz silme.
2. Bu belge ürünün ana gereksinim kaynağıdır. `docs/DECISIONS.md`, `docs/PROGRESS.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/EXPERIMENTS.md` dosyalarını oluştur; anlamlı her oturum sonunda güncelle.
3. **Bir seferde sadece bir milestone veya küçük dikey dilim** uygula. İlk oturum hedefi `M0 — Feasibility & Bootstrap` ve gerekiyorsa `M1`'in ilk veri prototipidir. Bütün fazları aynı anda üretme.
4. Bir bağımlılığın güncel API'sinden emin değilsen resmi dokümana veya yüklü sürümün kaynak koduna bak; yöntem ve argüman uydurma. Kurulum yapamıyorsan açıkça belirt.
5. Gerçek F1 verisini, simülasyonla üretilen veriyi ve editoryal/persona varsayımlarını **veri modelinde ayrı kategoriler olarak** tut.
6. Kodda tanımlanan her simülasyon davranışı için otomatik test yaz. Özellikle determinism, save/resume, branch isolation, legality, no-lookahead ve driver-style-conditioning testlerinden taviz verme.
7. İlk andan itibaren pause/save/load/branch mimarisini tasarla ve en küçük dikey prototipte çalıştır. Bunları bitmiş sisteme sonradan yamama.
8. **Fizik motorunu sıfırdan geliştirme.** Mevcut lap/sector-level yarış hesaplayıcılarından faydalan, gerekiyorsa küçük bir adapter yaz. Kullanıcı modelini direksiyon/gaz/fren seviyesinde eğitmek istemiyor.
9. **Eğitim fazı 1 sürücü profilleriyle başlar.** Sürücülerin risk, tutarlılık, lastik koruma, ıslak zemin yeteneği vb. parametreleri öğrenmenin hem girdisinde hem ödül/önceliklendirme mantığında etkilidir. Eğitim bitince sonradan eklenmiş kozmetik etiketler kullanma.
10. Asıl öğrenilen şey önce **kendi yarışını optimize etmek**, sonra **rakip ve takım arkadaşıyla strateji**, en son **sosyal stratejidir**. Gelecek fazlara ait altyapı arayüzleri tanımlanabilir; gereksiz özellikleri şimdiden uygulama.
11. Bir modelin gerçekten öğrenmediği bir davranışı “AI öğrendi” diye anlatma. Kural tabanlı, parametrik, öğrenilmiş ve LLM tarafından üretilmiş mekanikleri arayüzde ve raporlarda ayırt et.
12. Otomatik eğitim hiçbir zaman aktif dünyanın kayıtlı durumunu kendi kendine değiştirmesin. Eğitilmiş yeni politika kullanıcı tarafından açıkça seçildiğinde **yeni dünya veya fork** üzerinde etkinleştirilebilir.
13. Windows yerel kullanımında PowerShell komutları veya platformdan bağımsız `uv` / Python / npm komutları sağla. Docker'ı zorunlu yapma.
14. Her iş sonunda şunları raporla: yapılanlar; değişen dosyalar; çalıştırılan komutlar; test sonuçları; bilinen eksikler; sıradaki tek önerilen adım.
15. Kullanıcının soru-cevap sürecini tekrar başlatma; bu belge kararları içeriyor. Yalnızca gerçekten bloke edici bir kimlik bilgisi, lisans veya donanım gereksinimi çıkarsa soru sor. Diğer belirsizlikleri makul şekilde varsay ve karar günlüğüne yaz.

**İlk Claude Code komutu (kullanıcıdan kopyalanabilir):**

```text
Read F1_ALTERNATE_REALITY_MASTER_PLAN.md completely. Treat it as the main PRD and technical roadmap. Start with milestone M0 only: audit data sources, test FastF1 for Monza 2024, evaluate whether TUMFTM/race-simulation can be wrapped in a pausable step-based adapter, set up the repository and tests, and implement a minimal deterministic simulation state that can pause, save, load, resume and fork. Do NOT build the entire application, train an RL model, or implement fictional social agents yet. Produce real runnable code, tests, concise docs, and a truthful milestone report. Preserve driver profiles as first-class inputs to the eventual Stage 1 training system. Follow each acceptance gate before moving on.
```

---

## 1. VİZYON VE ÜRÜN İLKELERİ

### 1.1 Tek cümlelik vizyon

Gerçek F1 yarışlarından ve sezonlarından başlayıp, kişilikleri farklı AI sürücüleri ve stratejistlerin kararlarıyla alternatif bir şampiyona tarihi üreten; 2D pist üzerinde izlenebilen; istenilen anda durdurulabilen, kaydedilebilen, yüklenebilen ve farklı kararlardan itibaren dallandırılabilen yaşayan F1 simülasyonu.

### 1.2 Temel oyuncu deneyimi

1. Uygulamayı aç; kayıtlı dünyalarını veya yeni bir tarihsel başlangıcı gör.
2. `2024 Monza — Historical Baseline` dünyasını aç.
3. Gerçek start sıralamasına, pist haritasına, sürücü profillerine, lastiklere, hava bilgilerine ve strateji paneline bak.
4. Yarışı `Play` ile başlat; 2D pistte hareketleri, yarış sırasını ve model kararlarını izle.
5. Herhangi bir anda `Pause` yap. Aynı noktayı `Save` olarak sakla.
6. Yarışa daha sonra, uygulamayı veya bilgisayarı kapatmış olsan dahi aynı simülasyon durumundan devam et.
7. `Fork timeline` ile farklı pit kararı ver; bu kararın olası sonuçlarını analiz et. Ana kayıt değişmeden kalsın.
8. Modelin kararı ile kullanıcının kararı ve tarihsel referans stratejisi arasındaki farkları karşılaştır.
9. Zamanla tek yarıştan sezona, takım yönetimine ve karakterlerin sosyal ilişkilerine geç.

### 1.3 Kesin ürün ilkeleri

- **Simulation-first, AI-second, visuals-third:** Önce güvenilir simülasyon, ardından karşılaştırılabilir öğrenme, sonra görsel iyileştirme. Buna rağmen ilk 4–5 haftada basit bir 2D izleme deneyimi sun.
- **Always pausable/persistable:** Dünyanın ana otoritesi backend'dir. UI sadece gösterir.
- **Reproducible counterfactuals:** Aynı snapshot + model sürümü + seed + komut dizisi → aynı sonuç (desteklenen deterministik CPU modunda).
- **Profiles are causal:** Sürücü profili kararları ve davranış dağılımını ilk fazdan etkiler.
- **No magical realism:** Gerçek dünyadaki gizli lastik/araç/strateji parametreleri elimizde değil. Tahmini parametrelerle kalibre edilen bir yaklaşık simülasyon geliştiriyoruz.
- **Observability:** Neden pit kararı alındığı, model güveni, girdiler, kısıtlar, alternatif eylemler görülebilmeli.
- **Versioned history:** Kurallar, kadrolar, pistler, araçlar, modeller, sezonlar ve veri kaynakları sürümlü.
- **Research honesty:** Alternatif sonuçlar ispatlanmış tarihsel gerçekler değil, modelleme varsayımlarına bağlı tahminlerdir.

### 1.4 Başarı ölçütleri

**MVP başarı:** Monza 2024 için yarış durumu tutarlı görünüyor, bir tur/segment ileri yürütülebiliyor, pause/save/load/fork bozulmuyor, sürücüye koşullu bireysel strateji politikası baselinelerle kıyaslanıyor, alternatif pit planı sonuç dağılımı çıkarılabiliyor.

**V2 başarı:** Birden fazla ajan oyuncunun müdahalesiyle birlikte sezon içinde rekabet ediyor, takım arkadaşı ve rakip davranışları gözlemlenebiliyor, politika geliştirmeleri benchmark'larda ölçülüyor.

**V3 başarı:** Sosyal ajanların ilişkileri ve kararları zaman içinde kaydediliyor, sosyal durum sportif ve mali sonuçları sınırlı/açıklanabilir nedensel kanallarla etkiliyor; bütün sezon kayıtlı kalıyor.

### 1.5 Açıkça kapsam dışı (erken fazlar)

- Tam CFD, aerodinamik, drivetrain, süspansiyon veya gerçek zamanlı direksiyon fiziği geliştirmek.
- Sıfırdan AAA 3D oyun motoru yapmak.
- Gerçek sürücülerin özel ilişkileri veya gerçek motivasyonları hakkında doğrulanmamış iddiaları olgu gibi üretmek.
- Gerçek F1 telifli videoları veya radyoları gelişi güzel kopyalamak.
- Tarihsel yarışın gizli gelecek olaylarını AI'ın girdisine sızdırmak.
- Her gerçek yarışı %100 doğru yeniden üretmeyi garanti etmek.
- Bir yandan oyuncunun aktif yarışını çalıştırıp arka planda model ağırlıklarını fark edilmeden değiştirmek.

---

## 2. ÜÇ AŞAMALI AI ÖĞRENME YOL HARİTASI (DEĞİŞMEZ SIRA)

### STAGE 1 — SELF-OPTIMIZATION / INDIVIDUAL RACING INTELLIGENCE

**Amaç:** Sürücü, kendi özelliklerini, aracını ve pist/koşul bilgisini kullanarak **kendi beklenen yarış performansını** iyileştirir. Henüz rakibin niyetini tahmin etmez veya sosyal pazarlık yapmaz. Trafik gibi dış etkiler çevrenin bir parçası olabilir; bilinçli taktiksel rakip modellemesi bu aşamada yoktur.

Öğreneceği veya kullanacağı şeyler:

- Pist özellikleri: uzunluk, sektör özellikleri, hızlı/yavaş viraj yoğunluğu, overtake zorluğu, pit-lane zaman kaybı; **hazır veriler ve kalibrasyon**, sıfırdan sürüş çizgisi hesaplaması gerekmiyor.
- Kuru/ıslak koşullarda farklı lastiklerin göreli performansı.
- Lastik ömrü, aşınma ve sıcaklığın olası etkileri (ölçülebilen/öğrenilebilen yaklaşık parametreler).
- Yakıt yükünün yarış içinde tur süresine etkisi; **2024 F1 yarışlarında yarış sırasında yakıt ikmali yapıldığı varsayılmayacak**.
- Sürücünün agresif/pürüzsüz tempo seçimi ve bunun lastik aşınması, hız, hata riski üzerindeki etkisi.
- Pit-stop zamanlama, lastik bileşiği seçimi, tek/iki-stop karşılaştırması, hava öngörüsüne dayanarak pit kararı.
- Pistteki grip/weather koşullarına göre strateji uyarlama; eksik bilgi ve belirsizlikle karar verme.
- Yasal eylemleri değerlendirme: lastik stokları, yarış kuralları ve duruma göre geçersiz eylemleri filtreleme.

**Esas gereksinim:** Sürücü profili eğitim başlamadan sisteme kayıtlı olacak. Aynı çevrede aynı eğitim bütçesiyle `risk_pref=0.9` ve `risk_pref=0.2` gibi farklı vektörler test edildiğinde, mantıklı bir risk–tempo–aşınma farkı gözlenmeli. Aynı sürücü farklı koşullarda birebir aynı karara zorlanmamalı.

**Politika mimarisi:** Önce basit kurallar → optimize edilen Monte Carlo/MPC baseline → supervised pace/outcome predictor → sürücü vektörüyle koşullanan RL/imitasyon politikası. İlk günden RL eğitmek zorunlu değil; ilk eğitimden itibaren profiller kullanılmalı.

**Kontrol otoritesi:** `driver_pace_intent` bireysel sürücü politikası, `pit_timing/tyre_choice` ise Stage 1'de birleşik **driver-specific strategy policy** tarafından önerilir. UI bunu “sürücü+stratejist strateji kararı” olarak gösterir. V2'de takım otoritesi ile sürücü tercihleri ayrılır.

### STAGE 2 — MULTI-AGENT COMPETITION & COOPERATION

**Amaç:** Sürücü sadece kendi hızını optimize etmekle kalmaz; diğer sürücüleri ve takımları tahmin eder, pozisyon savunur, uygun anlarda undercut/overcut kovalar, takım arkadaşının yarışına değer verir.

- Opponent modeling: başka sürücünün pit zamanlaması, lastik davranışı, risk toleransı, yarış içi örüntüleri.
- Attack/defend/cooperate: geçiş denemesi, savunma seviyesi, takım emri, pozisyon verme, DRS/çekiş etkisi (kurala/sezona göre).
- Multi-agent environment: bir ajan değişince diğer ajanların stratejileri de yeniden hesaplanır.
- Team reward vs individual reward: pilot puanları / takım puanları / sürücü hedefleri arasındaki denge.
- İletişim: stratejist önerisi, sürücü tepkisi, takım arkadaşının katkısı.
- Stationarity sorunu: diğer ajanlar da öğrenirken politika karşılaştırmalarında sabit rakip havuzu ve geçmiş politika snapshot'ları kullan.

### STAGE 3 — SOCIAL / ECONOMIC / NARRATIVE INTELLIGENCE

**Amaç:** F1 dünyasının sportif ve sosyal tarihinin birikmesi.

- Kalıcı kimlik, hafıza, profesyonel ilişkiler, takım içi güven, rekabet gerilimi, takım moral durumu.
- Sözleşme / transfer / takım yönetimi / geliştirme bütçesi / basın açıklaması / müzakere.
- Ajanların karaktere uygun ancak gelişebilir, kuralları ihlal etmeyen uzun vadeli kararları.
- Sosyal olayların sportif etkileri sınırlı ve izlenebilir mekanizmalarla aktarılır (örnek: güvenin takım emrine uyma olasılığını etkilemesi).
- LLM kullanılabilir ama **LLM anlatı üretir ve sınırlı eylem önerir**; gerçek dünya durumuna serbest erişip keyfî veri değiştiremez. Bütün işlemler `SocialActionValidator` üzerinden geçer.

**Faz kapıları:** Stage 2 başlamadan Stage 1 benchmark + save/load/fork yeşil olmalı; Stage 3 başlamadan Stage 2 çoklu ajan davranışları test edilmiş olmalı.

---

## 3. SÜRÜCÜ PROFİL SİSTEMİ — AI EĞİTİMİNDEN ÖNCE HAZIRLANACAK

Bu bölüm projenin temel taşıdır. **Persona etiketleri UI süsü değildir; eğitim başından beri modele girer ve simülasyon geçişlerini/ödül fonksiyonunu etkiler.**

### 3.1 Profilin üç katmanı

1. **Immutable-ish seed / `DriverArchetype`:** Belirli sezon başında tasarlanmış başlangıç profili. Kimlik ve bilinen tarihsel takım bilgileri ayrı; sayısal sürüş kişiliği ise çoğunlukla tasarımcı tarafından belirlenen **tahmini, kurgusal simülasyon parametreleri**. Her biri `source_type: editorial_assumption` ile etiketlenir.
2. **Dynamic condition / `DriverCondition`:** O anda değişebilen yorgunluk, güven, form, lastik/sıcaklık koşulları, takım uyumu. Bunlar gerçek tarihsel veri değilse açıkça simülasyon verisidir.
3. **Learned policy / `DriverPolicy`:** Eğitimle oluşan karar verme yeteneği. `policy_version`, `training_dataset_version`, `profile_schema_version`, `checkpoint_hash` içerir.

Bir sürücü başka takıma geçtiğinde kimlik sabit; araç seviyesi, takım stratejisi ve sosyal bağlam değişir. Learned policy otomatik sıfırlanmaz; yeni düzenlemelere adaptasyon mekanizması sürümlü tasarlanır.

### 3.2 Gerekli profil parametreleri (0.0–1.0 aralığı; hepsi aynı kavrama karşılık gelmez)

| Alan | Anlam | Stage 1'de etkisi |
|---|---|---|
| `race_pace_skill` | Temel yarış temposu beceri prior'ı | İdeal koşullarda beklenen performans dağılımı |
| `consistency` | Turdan tura performans oynaklığının düşüklüğü | Tur süresi varyansı / hata dağılımı |
| `risk_appetite` | Getiri için risk üstlenme tercihi | Pace vs spin/lock-up/mistake risk tercihleri |
| `tire_management` | Lastik ömrünü korumaya dönük beceri | Sürüş modu–aşınma dönüşümü |
| `wet_skill` | Islak koşula uyum prior'ı | Grip etkileri altında tempo kaybı ve hata olasılığı |
| `adaptability` | Yeni pist/koşullara uyum eğilimi | Koşula özgü keşif/öğrenme hızı prior'ı; doğrudan gizli gelecek bilgisi değil |
| `qualifying_skill` | Sıralama turu odaklı beceri | Gelecekte sıralama simülatörü; V1'de tarihsel grid kullanılır |
| `overtake_skill` | Rakip geçişlerinde etkinlik | Stage 2 |
| `defense_skill` | Pozisyon savunma becerisi | Stage 2 |
| `team_orientation` | Takım hedefine yatkınlık | Stage 2 ödül ağırlığı |
| `composure` | Baskı altında istikrar | Stage 2 ve Stage 3 |
| `ambition` | Uzun vadeli kazanma/şampiyonluk hedefi | Stage 3 |
| `sociability` | Sosyal etkileşim eğilimi | Stage 3 |
| `negotiation_skill` | Görüşme kabiliyeti | Stage 3 |

Ek parametreler (gerektiğinde): `fuel_saving_skill`, `recovery_from_mistake`, `preference_for_early_pit`, `temperature_sensitivity`, `confidence`, `fitness`, `experience`, `stress_resilience`.

**Dikkat:** `risk_appetite` (tercih) ile `mistake_rate`/`consistency` (beceri) farklı kavramlardır. Riskli sürüş seçmek otomatik olarak kötü pilot olmak değildir. Aynı şekilde yüksek `tire_management` yavaş pilot olma anlamına gelmez. Ayrı nedensel kanallar kullan.

### 3.3 Başlangıç için örnek profiller (kurgusal tasarım prior'ları)

**Aşağıdaki sayılar gerçek biyometrik/telemetrik ölçümler veya kanıtlanmış kişilik analizleri değildir.** Kullanıcının istediği farklı sürüş arketiplerini göstermek için önerilmiş, değiştirilebilir oyun/simülasyon değerleridir. Ürün bunu açıkça göstermelidir. Gerçek sürücüye yönelik kesin psikolojik niteleme veya özel hayat çıkarımı yapılmamalıdır.

| Örnek persona | race_pace | consistency | risk_appetite | tire_management | wet_skill | adaptability | Öngörülen simülasyon yaklaşımı |
|---|---:|---:|---:|---:|---:|---:|---|
| **Max Verstappen — assertive competitor** | .96 | .90 | .82 | .88 | .94 | .90 | Uygun fırsatta tempoyu zorlayabilen, riski kabul eden güçlü tempo profili |
| **Lewis Hamilton — smooth veteran** | .93 | .94 | .58 | .95 | .93 | .88 | Uzun stint, tutarlılık ve lastik korumaya yatkın profil |
| **Charles Leclerc — qualifying-oriented attacker** | .92 | .86 | .74 | .82 | .83 | .85 | Hızlı tempo ve daha cesur performans seçenekleri |
| **Lando Norris — balanced adaptive racer** | .91 | .89 | .67 | .86 | .86 | .92 | Dengeli karar + hava/yarış evrimine hızlı adaptasyon |
| **Oscar Piastri — calculated competitor** | .90 | .92 | .62 | .90 | .84 | .90 | Ölçülü risk ve kuvvetli uzun-vadeli tempo |
| **Fernando Alonso — tactical veteran** | .89 | .92 | .70 | .91 | .89 | .94 | Değişen koşullarda strateji esnekliği |
| **Carlos Sainz — analytical stabilizer** | .89 | .90 | .60 | .88 | .83 | .89 | Planlı tempo ve stratejiye kontrollü uyum |
| **George Russell — assertive precision** | .89 | .87 | .73 | .85 | .86 | .89 | Tempoya ve pozisyona odaklı cesur tercih |

Bu tablo **2024 Monza'ya birebir resmi puanlar değildir**. 2024 start kadrosunun kalan sürücülerini de `seed_2024_monza_drivers.yaml` içinde tanımla: en az 20 ayrı kimlik ve profil, eksik olanlar için `generic_rookie`, `generic_veteran`, `generic_balanced` şablonlarından başlayıp `estimated=true` etiketi koy. İşlem `docs/ASSUMPTIONS.md` dosyasına kaydedilsin. `Hamilton = Mercedes`, `Verstappen = Red Bull` gibi sezon-özgü eşlemeler `season_roster_2024.json` dosyasında; 2026 için otomatik kullanılmasın.

### 3.4 Kaynağına göre özellik ayrımı

- `historical_verified`: Ad, start numarası, takım, start/grid, pist, yarış sonucu, tur ve pit olayları gibi doğrulanabilir kayıtlar.
- `estimated_from_data`: Pace baseline, degradasyon eğrisi, grip etkisi ve modelden türetilmiş tahminler.
- `editorial_assumption`: Risk, tutarlılık, kişilik, hırs, sosyal hedef gibi tasarım başlangıç parametreleri.
- `simulated_dynamic`: Simülasyon ilerledikçe oluşan yorgunluk, moral, ilişkiler, geçmiş kazançlar.
- `learned_parameter`: ML eğitimi sonucu güncellenen politika parametreleri ve varsa kişiye özel latent embedding.

Her profil parametresine mümkünse `value`, `source_type`, `confidence`, `schema_version`, `notes` eşlik etsin. Kullanıcı daha sonra editörden profilleri değiştirebilsin; değişiklik yeni bir `profile_revision` üretip timeline fork gerektirsin.

### 3.5 Örnek profil dosyası

```yaml
schema_version: 1
season: 2024
driver_id: "max_verstappen_2024"
identity:
  display_name: "Max Verstappen"
  car_number: 1
  team_id: "red_bull_2024"
  source_type: historical_verified
archetype:
  race_pace_skill: 0.96
  consistency: 0.90
  risk_appetite: 0.82
  tire_management: 0.88
  wet_skill: 0.94
  adaptability: 0.90
  qualifying_skill: 0.94
  overtake_skill: 0.92
  defense_skill: 0.92
  team_orientation: 0.70
  composure: 0.88
  ambition: 0.96
  sociability: 0.55
  negotiation_skill: 0.50
metadata:
  numeric_values_source_type: editorial_assumption
  estimated: true
  explanation: "Illustrative simulation persona; not verified driver psychometrics."
  profile_revision: "v1"
policy:
  type: "profile_conditioned_baseline"
  version: "untrained"
```

### 3.6 Profilin gerçekten Stage 1 eğitimini etkilemesi

**İki farklı mekanizma kullan:**

**(A) Simülasyon geçişleri:**

- `pace_mode` (conserve / balanced / push) temel hızı, lastik aşınmasını ve hata riskini değiştirir.
- `tire_management` yüksekse aynı `pace_mode` altında daha düşük tahmini aşınma eğilimi vardır; etkisi kalibrasyonla sınırlı tutulur.
- `consistency` rastlantısal tur sapmasının varyansını etkiler.
- `wet_skill` ıslak koşullarda performans kaybı dağılımını etkiler.
- `race_pace_skill` sürücünün baseline pace seviyesini etkiler, ancak **araç gücü ile sürücü becerisi ayrı değişkendir**.

**(B) Politikadaki tercih/ödül yapısı:**

- Politika girdisi `obs` içinde tüm ilgili profil vektörü yer alır.
- Sürücünün `risk_appetite` parametresi riskli strateji seçmenin beklenen fayda/ceza ağırlıklarını etkileyebilir. Ancak bütün sürücülere aynı genel yarış ödülünü de raporla: aksi halde farklı ödüllerle karşılaştırılan sonuçlar yanıltıcı olur.
- `profile-prior regularization`: Öğrenilen davranışlar sürücü prior'ından çok uzaklaşırsa hafif regularization uygula; zorunlu sabit aksiyonlara dönüşmesin.
- Eğitim tek bir sürücüye ayrı ayrı model eğitmek zorunda değil: **ortak profile-conditioned policy** `pi(a | s, p_driver, p_car, p_track)` tüm sürücülere bilgi aktarabilir. Daha sonra sürücüye özel embedding, adapter veya fine-tuning deneyebilirsin.

**Politika farkı testleri:** Aynı senaryo, araç, rastgelelik ve gözlem altında (1) Max tarzı profili, (2) Hamilton tarzı profili, (3) tamamen dengeli profili koştur. Eylem dağılımı, stint uzunluğu, temponun değişkenliği, aşınma ve beklenen yarışı bitirme süresi karşılaştır. Sonuçlar aynılaşmışsa profil kullanılmıyor veya model tüm profilleri aynı stratejiye indirgemiş olabilir.

### 3.7 Profil–performans adaleti ve ince ayar

- Bir sürücüye yüksek pace, yüksek konsistensi ve yüksek lastik koruması vermek, haksız biçimde her metrikte üstün karakter yaratabilir. Başlangıçta **dengeleyici yetenek bütçesi** veya oyun dengesi analizi önerilir, ama gerçek sürücü performansının yapay olarak eşit olması şart değildir.
- Kalibrasyonda sadece Monza sonuçlarına uyum sağlama; farklı pist ve hava koşullarında makul genelleme aranır.
- `style` ile `skill` farklı raporlanır. Kazanma olasılığı yüksek bir sürücü mutlaka agresif olmak zorunda değildir.
- Belirsiz/niteliği tartışmalı gerçek sürücü profilleri “editoryal başlangıç profili” olarak arayüzde düzenlenebilir olmalı.
- Her deneme `profile_revision` ve `model_version` kaydı tutmalı.

---

## 4. CANLI DÜNYA: DURDURMA, KAYDETME, DEVAM ETME VE ALTERNATİF TARİH

Bu özellik **MVP'den itibaren zorunlu**. Sonradan eklenecek bir 'Save button' değildir.

### 4.1 Üç ayrı zaman sistemi

1. `wall_clock`: Bilgisayarın gerçek saati. Uygulama kapalıyken bile ilerleyebilir; **dünya kapalıyken otomatik ilerlemez** (isteğe bağlı asynchronous world mode gelecekte ayrı özellik).
2. `simulation_clock`: Yarış içinde ilerleyen otoritatif oyun zamanı. Pause halinde tamamen donar.
3. `presentation_clock`: Frontend'in 30/60 FPS render/interpolasyon zamanı; oyun mantığını değiştirmez.

Ayrıca `championship_calendar` sezonlar arası günleri ve sosyal olayları temsil eder. Yarış esnasında `race_sim_seconds`, `lap`, `sector`, `distance_along_track_m` gibi değerler vardır. Sezon ilerlemesi duraklatılmış yarışla birbirine karıştırılmamalıdır.

### 4.2 Durum makinesi

```text
CREATED --start--> RUNNING --pause--> PAUSED --resume--> RUNNING
   |                   |                |                |
   +--save------------> |                +--save----------+
                       +--finish--> COMPLETED
                       +--error----> ERROR_RECOVERABLE

PAUSED --fork--> NEW WORLD (PAUSED, copied state + separate history)
PAUSED --load another save--> separate loaded world/session
COMPLETED --fork from earlier snapshot--> alternative world
```

`RUNNING` iken `save` isteği alınabilir, fakat snapshot **tutarlı simülasyon tick'inin sonunda** alınır; UI'da görüntülenen son onaylı tick gösterilir. `Pause` isteği yeni bir geçiş başlamadan kuyrukta işlenir. Kullanıcının “istediğim an durdurabilmeliyim” gereksinimi teknik olarak bu şekilde karşılanır; izin verilen maksimum tepki gecikmesi MVP'de hedef <250ms UI bildirimi ve 1 simülasyon-tick güvenli sınırıdır. Eğer kullanılan fizik/race adapter'ı sadece bütün turu tek seferde ilerletebiliyorsa bu gecikme sağlayamayabilir: bu durumda adapter **küçük sektör/olay adımlarına ayrılmalı** veya daha granular bir motora geçilmelidir. 'Anında durdurma' iddiası tutarsız tick veya uzun bloklayan tur adımıyla yapılmamalıdır.

### 4.3 Simülasyon tick'leri ve animasyon

- Simülasyon otoritesi `WorldSimulation.step()` işlevidir; o anki snapshot + komutlar + kontrollü RNG ile **tek belirlenmiş geçiş** üretir.
- MVP için `segment` veya 0.25–1.0 simülasyon saniyesi gibi küçük ve stabil canonical step seç (performans testleri sonucu kararlaştırılır). Sektör/segment olayları ve tur sonunda strateji hesaplamaları tetiklenir.
- Gerçek fizik simülasyonu zorunlu değil: `RacePaceEngine` hesapladığı beklenen tur süresini pist uzunluğuna dağıtan, segment farklılıklarını basit/kalibre edilmiş katsayılarla kullanan adapter olabilir. Bu **fiziksel mikro-sürüş simülasyonu değildir**.
- `x1, x2, x5, x10, x25`, `Pause`, `Step Tick`, `Step Segment`, `Step Lap` kontrolleri. Render FPS bağımsız.
- Çok yüksek fast-forward modunda daha seyrek render yap; aynı canonical state transitions ve RNG sırasını koru.
- Frontend interpolasyon yapsa bile `paused` olduktan sonra UI konumu son canonical state'te sabitlenir. UI kendi araç/yarış state'ini uyduramaz.

### 4.4 Otoritatif dünya durumu: `WorldState`

Kaydedilen dünyanın özeti aşağıdaki tüm kategorileri içermelidir:

```json
{
  "schema_version": 1,
  "world_id": "world_001",
  "timeline_id": "timeline_main",
  "parent_timeline_id": null,
  "branch_point_event_id": null,
  "world_status": "PAUSED",
  "mode": "HISTORICAL_ALTERNATE",
  "season_id": "2024",
  "race_id": "2024-monza-race",
  "regulations_version": "fia_2024_v1",
  "dataset_manifest_hash": "sha256:...",
  "engine_version": "race_engine_v0.1.0",
  "active_policy_manifest": {"strategy": "policy_baseline_v1"},
  "profile_manifest_hash": "sha256:...",
  "sim_tick": 920,
  "simulation_clock_seconds": 1263.5,
  "calendar_datetime": "2024-09-01T13:21:03.500Z",
  "rng_states": {"python": "...", "numpy": "...", "torch": "..."},
  "cars": "<serialized CarState objects>",
  "tyre_allocations": "<per-car legal available sets>",
  "race_control": "<flags, penalties, neutralization>",
  "weather_state": "<current and observed forecast>",
  "driver_conditions": "<dynamic state>",
  "team_states": "<current strategy, resources>",
  "social_state": "<future schema, empty initially>",
  "command_cursor": 118,
  "event_cursor": 472,
  "saved_at_wall_clock": "2026-10-10T00:00:00Z"
}
```

Bu **yalnızca şema örneğidir**. `cars` gibi alanlar gerçek kodda geçerli tipli JSON/Pydantic nesneler ve/veya normalize tablolar olmalı; `rng_states` basitleştirilmiş string gösterimi yerine taşınabilir ve doğrulanabilir binary/JSON codec ile saklanmalı. Dosyada JSON olarak yer alamayacak nesneleri `pickle` ile kontrolsüz yükleme; güvensiz deserialization'a açık kapı bırakma.

**Kesin kayıt içeriği:** pozisyon, tur, sektör, araç/sürücü kimliği, lastik bileşiği/ömrü, stokları, pit durumu, beklenen pace, yakıt/tahmini enerji, sıcaklık ve hava, pist grip durumu, lastik aşınma state'i, takım stratejisi, tüm RNG durumları, her ajanın politika kimliği/sürümü, sürücü profil revizyonu, hasar/ceza/yarış kontrol durumları, dünya/şampiyona olayları, social memory (sonraki faz).

### 4.5 Snapshot + event log + branch (önerilen hibrit)

**Snapshot:** Belirli tick'teki tam state. İlk MVP'de SQLite + dosya sistemi kullan.

**Event log:** Zaman sıralı `pit_requested`, `pit_executed`, `tyre_changed`, `overtake`, `weather_changed`, `strategy_decided`, `user_override`, `race_finished`, `social_event` gibi olgular ve kullanıcı komutları. Olaylar aynı event clock üzerinde toplam sıralanır.

**Branch:** Yeni dünya aynı snapshot'tan başlar ama kendi `timeline_id`, RNG akışı, komut/event geçmişi ile devam eder. Kaynak snapshot immutable kalır. `common_random_numbers` modu iki branşın exogenous hava/hata şoklarını ortak seed schedule'dan çekebilir; eyleme bağlı rastgele sayı tüketiminde kaymayı önlemek için `event_key`-tabanlı bağımsız RNG stream'leri kullan.

**Replay garantisi:** (a) aynı engine, dataset, profile, policies, events ve RNG ile state replay; (b) video 'historical replay' ile canlı alternatif simülasyon birbirine karışmasın. RNG durumunun kaydı gerekli ama bazı BLAS/GPU/parallel işlemlerinde bitwise eşitlik garanti edilemez. Deterministik CPU modunun kapsamını belgelerle sınırla.

### 4.6 Transaction ve veri dayanıklılığı

- SQLite `WAL` (write-ahead log) + foreign keys + işlemsel snapshot metadata.
- Snapshot içeriğini geçici dosyaya yaz, checksum üret, `fsync`/atomik rename (desteklenen sistemlerde), sonra DB manifest referansını işlemsel olarak etkinleştir. Alternatif: state blob + manifest'i tek SQLite transaction'ında sakla.
- Tamamlanmamış/bozuk snapshot'ı yükleme; checksum ve schema version doğrula.
- Autosave: her kritik komut ve her `N` sim tick / X dakika (varsayılan kullanıcı tarafından değiştirilebilir). Önceki sağlam kayıt korunur.
- Kayıt sırasında uygulama kapanırsa eski snapshot güvenle yüklenir. `recoverable_error` göster.
- DB migrations (`schema_version`), eski kayıtları yükleme veya migrasyon prosedürü; sessiz state kaybı kabul edilemez.
- Çift tıklanan komutlar için `idempotency_key`. Aynı komut iki kez pit yaptırmamalı.
- V1'de yerel, tek kullanıcı ve tek authoritative simulation worker. Eşzamanlı writer yok.

### 4.7 Zorunlu pause/save/fork kabul testleri

1. `state_A` kaydet, uygulama nesnesini tamamen yok et, yeni süreçte yükle → önemli bütün state alanları eşit.
2. Aynı seed ve modelle save olmadan 1000 tick vs 400 tick+save/load+600 tick → bit düzeyi desteklenen CPU determinism modunda `hash(state)` eşit.
3. Pause sonrası `tick`, lastik aşınması, dünya tarihi, event sayısı ve bütün araç koordinatları değişmez.
4. Pause → 5 saniye gerçek süre bekle → resume: yarış zamanı yalnızca resume sonrasında ilerler.
5. Fork A→B; B'de başka pit kararı; A değişmez, B farklı sonuç üretebilir.
6. Branch'in eski snapshot'ını silmek/yeniden adlandırmak mevcut branşları bozmaz (referans sayımı veya bağımsız manifest/özdeş blob).
7. Bozuk snapshot checksum'u reddedilir, son sağlam autosave önerilir.
8. Uygulama kapanırken aktif dünya otomatik güvenli duruma getirilir veya son başarılı autosave korunur.
9. Aynı komut `idempotency_key` ile iki kez geldiğinde ikinci defa uygulanmaz.
10. Kullanıcı tam event sınırında (`pit_request`, `overtake`) pause yaptığında kayıt tutarlı kalır.

---

## 5. HISTORICAL REPLAY, ALTERNATIVE SIMULATION VE EVREN MODLARI

| Mod | Veri kaynağı / sürüş | Çıktı | Ne anlama gelmez? |
|---|---|---|---|
| `HISTORICAL_REPLAY` | Kaydedilmiş gerçek olayları ve pozisyonları mümkün olduğunca izler | Geçmiş yarışı izleme | AI'ın alternatif geleceği üretildiği anlamına gelmez |
| `HISTORICAL_ALTERNATE` | Gerçek start + o ana kadarki gözlemler → ayrışma anından sonra simülasyon | Alternatif yarışı hesaplama | Sonucun tarihsel kesinlikle öyle olacağı anlamına gelmez |
| `FREE_SIMULATION` | Simüle başlangıç, hava, sürücüler, araçlar, stratejiler | Serbest yarış | Resmi gerçek yarış kaydı değildir |
| `CHAMPIONSHIP` | Sezon takvimi + yarışlar + kalıcı ekonomi/sosyal state | Yaşayan alternatif F1 tarihi | Kamuya açık gerçek transfer/ilişki kaydı değildir |

**Divergence contract:** Kullanıcı `Lap 18`'de farklı bir pit kararı verirse `Lap 1–17` tarihsel/kanonik olabilir. Divergence sonrası rakip pozisyonları ve pit kararları eski kayıt dosyasından kopyalanmamalıdır: yarış motoru yeniden hesaplamalıdır. `historical_reference` ayrı bir çizgi olarak gösterilebilir.

### 5.1 Monza 2024 referans olayı

2024 Monza resmi sonucunda Leclerc kazandı; Piastri ikinci ve Norris üçüncü bitirdi. Referans yarış 53 tur sürdü. Bu event baseline kalibrasyonu için iyi bir seçimdir. Bu veriler resmi sonuç veya doğrulanmış timing kaynaklarından okunacak; dosyada sabit kodlanmış bir "kazanan" gerçek alternatif simülasyona dayatılmayacak.

### 5.2 Sezon kuralları ve kadro bağlamı

- 2024 ruleset, 2026 ruleset ve gelecekteki alternatif kurallar ayrı `RegulationSet` şemalarıdır.
- DRS ve diğer overtaking/recharge mekaniklerini sezona göre düzenle. Özellikle 2026 teknik kuralları 2024'ten ciddi şekilde farklıdır; modern 2026 takım/araç/elektrik sistemi mantığını 2024 Monza'ya yanlışlıkla uygulama.
- Kuru havada birden fazla slick compound kullanımı gibi kurallar, yarışın ilgili yılının **güncel yürürlükteki sportif düzenlemesinden** doğrulanarak eylem validator'ına eklenmeli; istisnalar (ıslak yarış vb.) doğru uygulanmalı.
- 2024 boyunca sürücü transferlerinin zaman içinde olması nedeniyle sezon değil **yarış başına roster snapshot** kullanılmalı; Monza'daki sürücüleri yılın başka yarışlarına otomatik taşıma.
- Takım güçlü/zayıf yanları `car_performance` katmanında; sürücü yeteneğiyle karışmasın.

---

## 6. VERİ KAYNAKLARI, GERÇEKLİK SINIRLARI VE LİSANSLAR

### 6.1 Doğrulanmış başlangıç kaynakları (10 Ekim 2026 itibarıyla)

| Kaynak | Ne sağlar? | Not / uygulama |
|---|---|---|
| **FastF1** — https://docs.fastf1.dev/ | Tur süreleri, yarış sonuçları, telemetri, position, lastik, hava ve oturum metadata | `get_session` / `session.load` + yerel cache. Her event için veri kapsamı değişebilir. MIT kod lisansı veriye sınırsız ticari kullanım hakkı vermez. |
| **OpenF1** — https://openf1.org/docs/ | 2023+ geçmiş oturumlar, car data, positions, laps, stints, weather, pit, race control | Kişisel/eğitim/non-commercial kullanım; ücretsiz tarihsel erişim; 3 req/s ve 30 req/min planı doğrulandı. Konum verisi ideal fizik ölçümü değildir. |
| **Jolpica-F1** — https://github.com/jolpica/jolpica-f1 | Geçmiş sonuçlar, sürücüler, takımlar, takvim, grid, pit gibi Ergast-benzeri endpoint'ler | Ergast 2025'te kapandı; bugün Jolpica kullan. Rate limit ve TOS kontrol et. |
| **TUM Race Simulation** — https://github.com/TUMFTM/race-simulation | Tur bazlı yarış/strateji simülasyonu, tyre degradation, fuel effect, traffic, Monte Carlo, örnek RL VSE | Python 3.8 döneminde geliştirilmiş, önceden hazırlanmış yarış parametreleri 2014–2019. Güncel Python/Monza 2024 ile uyumluluk ve step/save adapter'ı doğrulanmalı. **LGPL-3.0**. |
| **TUM Racetrack Database** — https://github.com/TUMFTM/racetrack-database | Monza vb. pistlerin XY centerline, width ve raceline koordinatları | Harita doğruluğu pistten piste değişebilir. **LGPL-3.0**. |
| **TUM F1 Timing DB** — https://github.com/TUMFTM/f1-timing-database | 2014–2019 F1 timing parametreleri | **CC BY-NC-SA 3.0**, 2024 için hazır kalibrasyon değil. |
| **F1 official results** — https://www.formula1.com/en/results/2024/races/1244/italy/race-result | Monza 2024 tarihsel nihai sonuçları | Karşılaştırma için; resmi içeriği yeniden dağıtmadan önce haklar kontrol edilmeli. |
| **FIA regulations** — https://www.fia.com/F126 | 2026+ için resmi kural kaynakları | Tarihsel yılların kendi kural dökümanlarını ayrıca sakla. |

OpenF1 web sitesi verilerin eğitim/kişisel/non-commercial kullanımına yönelik olduğunu belirtir; OpenF1'i kullanmak resmi F1 yayını veya görsellerini çoğaltma hakkı vermez. F1 ve takım logoları/ticari markalar ayrı kontrol edilmelidir. TUM projelerinin **kod** lisansları ve bağlı **veri** lisansları birbirinden farklı olabilir. Bunları `docs/LEGAL_AND_LICENSES.md` içinde açıkça izle. Tek kullanıcı/araştırma MVP'si için tasarla; ticari yayına çıkmadan yeniden lisans değerlendirmesi yap.

### 6.2 İndirilecek minimum veriler

`session_metadata`, `drivers`, `teams`, `start_grid`, `laps`, `stints`, `pit_events`, `weather`, `track_status`, `race_control`, `results`, `position_samples`, `car_telemetry (isteğe bağlı)`, `track_geometry`.

Her dataset şu manifest bilgilerini taşımalı:

```yaml
dataset_id: monza_2024_v1
event: 2024_italian_gp
source: fastf1
source_api_version: "<record actual>"
retrieved_utc: "<record actual>"
session_key: "<record actual>"
license_review_status: "research_only"
files:
  - relative_path: data/processed/monza_2024/laps.parquet
    sha256: "<computed>"
    rows: "<computed>"
unit_conventions:
  time: seconds
  temperature: celsius
  distance: meters
missingness_notes: "<populate>"
```

**Data model prensibi:** her değere mümkünse `observed`, `derived`, `estimated`, `synthetic`, `missing` etiketleri ve `source_id` eşlik etsin. Tahminle doldurulan gerçek yakıt miktarı veya grip değeri ham veriden gelmiş gibi gösterilmesin.

### 6.3 Gerçek verilerde görünmeyenler

- Gerçek takımların özel lastik modelleri, tam araç ayarları, gerçek pit çağrısının bütün gerekçeleri, sürücülerin iç motivasyonları, takım özel strateji yazılımları, gerçek zamanlı yakıt/sıcaklık detayları genellikle erişilebilir değildir.
- Dolayısıyla bu parametreleri veri destekli yaklaşık modeller veya makul senaryo prior'larıyla tahmin etmek gerekir.
- Tahmin edilen her parametre için belirsizlik aralığı veya senaryo duyarlılık testi hazırlanmalı.
- **Gerçek lastik sıcaklığı** ve **hava sıcaklığı** birbirinin yerine kullanılmamalıdır.
- Hava tahmini kullanılıyorsa karar anındaki yalnızca erişilebilir hava gözlemleri ve gerçekçi forecast kullanılmalı; tüm yarışın gerçek geleceği input olamaz.

### 6.4 Veri kalitesi pipeline'ı

1. `extract`: API çağrılarını retry/backoff/rate limit/cache ile indir.
2. `validate_raw`: event/session ID, dtypes, timezone UTC, duplicate, order, negative time ve önemli boşluklar.
3. `normalize`: driver/team/race stable IDs, unit conversion, pit-lap edge cases, stint boundaries.
4. `derive`: lap pace features, weather-at-lap, tyre age, traffic proxy, gap/interval, sector weight.
5. `annotate`: eksik/tahmini alanların kalite bayrakları.
6. `freeze`: `dataset_manifest` ve hash, immutable experiment dataset.
7. `audit`: official results ile start/finish consistency, off-by-one tire age, pit lap convention.

### 6.5 Örnek FastF1 araştırma komutu (API kullanımını kurulu sürümde doğrula)

```python
from pathlib import Path
import fastf1

Path(".cache/fastf1").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(".cache/fastf1")
session = fastf1.get_session(2024, "Italy", "R")
session.load()
print(session.event["EventName"])
print(session.laps[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]].head())
print(session.weather_data.head())
```

Bu örnek **fetch doğrulama başlangıcıdır**, hazır bir üretim pipeline'ı değildir. Eksik alan, API format farkı, NaN ve download hataları için test gereklidir. `session.load()` sırasında bütün telemetriyi hemen çekmek yerine ihtiyaca göre minimal yüklemenin mümkün olup olmadığını güncel FastF1 dokümanından kontrol et.

---
## 7. RACE ENGINE: KENDİ FİZİĞİMİZİ YAZMADAN GERÇEKÇİ BİR YARIŞ

### 7.1 Fizik ve yarış hesaplama sınırımız

Kullanıcı doğrudan araç kontrolü veya diferansiyel denklem tabanlı lastik/aerodinamik fiziği yazmak istemiyor. Buna rağmen **stratejik açıdan fiziksel olarak makul** bir simülasyon gerekli. V1 hesaplayıcısı şu soyutlama ile çalışabilir:

- `lap_time_base(driver, car, track, lap)` gerçek historical training laps'tan türetilir.
- `tyre_degradation(compound, age, driver_style, track, temp)` veriyle kalibre edilmiş fonksiyon/predictor olarak uygulanır.
- `fuel_mass_effect(remaining_laps)` pozitif ve yarış ilerledikçe azalan yaklaşık penalty (2024 yarışında pit yakıt ikmali yok).
- `weather_grip_modifier(rain, surface_wetness, compound)` koşul etkisi. Yağmur, lastik seçimi ve pist grip'i birbirine bağlanır.
- `traffic_effect`, `dirty_air_effect`, `overtake_cost`: önce kontrollü yaklaşık modeller, sonra data-driven iyileştirme.
- `driver_pace_intent`: conserve / balanced / push gibi kararların beklenen tur süresine, lastik tüketimine ve hata olasılığına etkisi.
- `stochastic_terms`: zamanla korelasyonlu, tutarlılığa ve şartlara bağlı ama sınırlandırılmış noise ve reliability olayları.
- `pit_loss`: pitlane transit + stop + giriş/çıkış ve trafik; kaynak veri uygun değilse kalibre edilen dağılım.
- `safety_car`, `VSC`, `yellow`, `red`: Stage 1'de basitleştirilmiş ve açık varsayımlar; zamanla daha ayrıntılı.

Basit kavramsal formül:

```text
expected_lap_time = baseline_car_driver_track
                  + tyre_age_effect
                  + fuel_mass_effect
                  + weather_and_surface_effect
                  + traffic_and_track_status_effect
                  + driver_pace_mode_effect
                  + estimated_uncertainty
```

Bu bir **modelleme şemasıdır**, doğrudan gerçek dünya fizik kanunu veya her etkinin toplanabilir olduğunun garantisi değil. Etkileşimler gerekiyorsa çarpımsal/multivariate modelle değiştir. Negatif/sıfır tur süresi, imkânsız hız ve uçuk istatistikler validator tarafından reddedilsin.

### 7.2 Simülatör seçiminde karar kapısı

`M0` sırasında üç alternatifi test et:

**A. TUM Race Simulation adapter (ilk aday)**

- Artısı: race strategy, tyre, fuel, traffic, Monte Carlo ve RL örneği içerir.
- Riski: Python 3.8 bağımlılıkları, 2014–2019 parametreleri ve API'sinin tam/orta-tur state checkpoint'e uygun olup olmaması.
- **Feasibility testi:** örnek yarış çalışıyor mu; Monza 2024 için veri parametreleri çıkarılabiliyor mu; deterministik `step` ve `get_state/set_state` arayüzüne **lisansı koruyarak** sarılabiliyor mu; yarış 100–1000 simülasyonda yeterince hızlı mı?

**B. Time-discrete / başka açık kaynak yarış motoru**

- TUM bağlantılarındaki zaman ayrık motorları incele. Lisans, bakım, ara state ve güvenilirlik bakımından değerlendir.

**C. Kendi minimal race orchestration katmanımız**

- Fizik motoru yazma! TUM'dan fikir/kullanım koşulları izinli bileşenler veya **mevcut data-driven lap-time predictor** üzerinde küçük deterministic event scheduler ve araç pozisyon interpolasyonu yaz.
- Bu fallback, yarışın **sıralama, tur zamanı, lastik ve pit stratejisi** kurallarını düzenler; direksiyon fiziği üretmez.

**Karar ölçütü (sıralı):** (1) pause/save/load/fork uygunluğu, (2) reproducibility, (3) 2024 kalibrasyonu, (4) performans, (5) lisans güvenliği, (6) bakım kolaylığı. En gelişmiş fizik simülatörü otomatik olarak en iyi seçim değildir.

### 7.3 `RaceEngine` portu

```python
from typing import Protocol

class RaceEngine(Protocol):
    def initialize(self, race_config, roster, profiles, scenario_seed): ...
    def get_state(self): ...                # canonical serializable snapshot
    def set_state(self, state): ...
    def get_observation(self, agent_id): ...
    def get_valid_actions(self, agent_id): ...
    def apply_commands(self, commands): ...
    def step(self, max_sim_delta_seconds: float): ...  # bounded, no hidden long loop
    def get_events(self, since_event_id): ...
    def get_positions(self): ...
    def is_terminal(self) -> bool: ...
```

`step` zorunlu olarak kısa bir otoritatif ilerleme üretir. TUM kütüphanesi bu sözleşmeye doğrudan uymuyorsa ince sarmalayıcı yetmeyebilir: adapter yerine fallback seç. **Pause/save/load/branch gereksiniminden vazgeçme.**

### 7.4 Yarış hesaplaması ve 2D görsel koordinatlar

- Pist koordinatları `racetrack-database` tarafından sağlanabilir: `centerline_points`, `raceline_points`, `cumulative_distance_m`, `track_widths`.
- Araç canonical durumu `distance_along_track_m` ve `laps_completed` olarak tutulur. Ekran X/Y noktası arc-length interpolasyonundan gelir.
- Yerleşik AI sürüş fiziği olmadığı için saniyelik araç hızını, kalibre edilen pace/segment modeli türetir; interpolasyonda araçların görsel olarak çarpışmaması için küçük lateral UI offset kullanılabilir ama bunun yarış sonucunu **değiştirmesine izin verilmez**.
- `overtake` sadece ekranda iki sprite'ın birbirini geçmesi değildir: race-engine pozisyon/sıralama olayı güncellemelidir.
- Pit-lane için ayrı çizgi ve durum makinesi (`ON_TRACK`, `PIT_ENTRY`, `PIT_LANE`, `PIT_BOX`, `PIT_EXIT`). Ekran sırf pit-stop animasyonu yaptığı için lastik değişmiş varsayma.
- Yol tutuşu pistin tek 'grip' sayısı olarak başlayabilir, sonra sektor-level `surface_grip`/`wetness` ve `rubbering_in` gibi katmanlara ayrılabilir; kaynak/kalibrasyon belirsizliğini göster.
- Pistlerin yüksekliği/süperelevasyon vb. veriler V1'de olmayabilir; fizik iddialarında bulunma.

### 7.5 Yarış olayları ve legalite

`RaceRules` / `ActionValidator` şu kontrolleri kapsamalı:

- Sürücü yarışta mı, DNF oldu mu, pit girişi mümkün mü?
- Mevcut/hazır lastik seti var mı, izin verilen bileşik mi, aynı set yeniden kullanılabilir mi?
- Yıl bazlı kuru hava bileşik zorunlulukları ve wet tyre istisnaları.
- 2024 için refuelling action yasak; yakıt **tüketilir** ama pitte **doldurulmaz**.
- Pistin geçiş/atak kuralı ve DRS gibi eylemler ancak ilgili yıl koşullarında geçerli.
- VSC/SC durumuna göre hız ve pit-time avantajları değişebilir.
- Pit için tek komut; geçersiz pit kararlarında motor güvenli default `stay_out` uygulayabilir ama nedeni event log'a yazılır.
- Yarış sıralaması lapped cars, DNF ve penaltılara göre tutarlı.

### 7.6 Gerçekçilik hedefi ve kalibrasyon

**Historical replay** için F1 veri setinden start/grid/finishing order ve temel olay eşlemeleri doğru olmalı. **Independent simulated baseline** için aynı sonucu birebir çıkarma zorunlu değil; anlamlı metrik toleransları belirlenir.

Ölçümler:

- `lap_time_mae_s`, `lap_time_rmse_s`: tahmin vs uygun held-out gerçek tur verisi.
- `pit_lap_error`, `stint_length_error`.
- `classification_accuracy` değil, `position_error` ve `rank_correlation` da raporlanır.
- `finish_gap_error`, `race_time_error`, `incident_frequency`.
- Driver/team/track subgroup metrics.
- Lastik aşınma ve weather sensitivity curves.
- Baseline reproducer testleri: test datasetlerinde model, eğitim verilerindeki tek pit kararlarını ezberlemiyor mu?

**Ön kalibrasyon hedefleri** gerçekçi araştırma hedefi olarak konulmalı, garanti değildir. Örneğin daha ilk hafta `lap_time_MAE < 1s` zorunluluğu koymak veri kalitesinden dolayı yanlış olabilir. Önce baselineleri ölç, ardından hedef toleransları `docs/EXPERIMENTS.md` içinde veriyle belirle.

---

## 8. ML MODEL 1: PROFILE-CONDITIONED PACE / DEGRADATION PREDICTOR

### 8.1 Görevi

Her araç/sürücü için beklenen tur süresi dağılımını ve lastik performans değişimini tahmin etmek. Sadece tek nokta tahmini değil, örneğin `p10/p50/p90` veya beklenen değer + varyans belirsizliği üretmek.

### 8.2 Girdiler

- `driver_profile_vector` (ilk gün, zorunlu).
- `car_performance_vector` (ayrı, takım/araç bazlı).
- `track_embedding` veya pist özellikleri (düzlük oranı, yavaş viraj payı, pit kaybı, degrade class).
- `tyre_compound`, `tyre_age_laps`, `tyre_set_history`.
- `fuel_remaining_estimate` (gerçek ölçüm yoksa prior/derived flag).
- `air_temp`, `track_temp`, `humidity`, `rain_intensity`, `surface_wetness_estimate`, `weather_confidence`.
- `pace_mode`, `recent_lap_pace`, `track_status`, `traffic_proxy`.
- `season_ruleset_id`, `observability_mask` ve eksik değer göstergeleri.

**No-leakage:** Laptaki sonuç tahmin edilirken o turun sonundaki lastik/aşınma ve hava değerleri yanlışlıkla feature olarak girmez. Her feature'ın `available_as_of_tick` bilgisi olsun.

### 8.3 İlk modeller

1. Baseline A: Aynı event'te geçmiş turlardan robust rolling median + compound/age düzeltmesi.
2. Baseline B: Linear/GAM veya gradient boosted trees (LightGBM/XGBoost).
3. Neural model: profile-conditioned MLP, tabular embedding + residual branch; gerekirse geçmiş turlar için GRU/Transformer.
4. Karşılaştırma: modeli profile feature'ları olmadan eğit, farkı ölç.

ML hedefini yarış motoruyla döngüye sokarken dikkat: Gerçek veride sürücü profili sayılarını biz tasarladık; bunları sanki tarihsel gerçek ground truth'muş gibi doğrulama. Model, genel observed pace ilişkilerini öğrenir; profil etkileri belli ölçüde önceden tanımlanan simülasyon prior'larından gelebilir. İleride veri destekli latent profillerle ilişki kurulur.

### 8.4 Veri bölme stratejisi

- Aynı yarışın komşu turlarını hem train hem test'e rastgele dağıtma; ciddi leakage olur.
- Önce `leave-one-race-out` / yarış bazlı veya zamansal split; yeni pistte genelleme için `leave-one-circuit-out` ek test.
- 2024 Monza'yı örnek gösteri yarışı yapıyorsak, mümkün olduğunca ayrı held-out yarışları performans kanıtı olarak kullan. Monza parametre kalibrasyonu için kullanılmışsa Monza üzerinde ölçülen sonuç **in-sample** diye etiketlenmeli.
- Telemetri verisi missingness ve örnekleme farkları için mask ve audit tutulmalı.
- Test senaryolarında dry / transition-to-wet / wet / SC / DNF alt gruplar.

### 8.5 Ölçümler ve deneyler

- MAE/RMSE/median absolute error, calibration, NLL veya pinball loss.
- Profil ablasyonu: `with_profile vs without_profile`.
- Koşul ablasyonu: `with_weather vs without_weather`.
- Team/car confounding kontrolü: sürücü farkını sadece araca yükleyen model olmamalı.
- Synthetic stress tests: artan sıcaklıkta lastik aşınması, beklenmedik yağmur, ani düşen grip.

### 8.6 Model manifesti

```yaml
model_name: pace_predictor
model_version: v0.2.0
artifact_sha256: "<computed>"
profile_schema_version: 1
feature_schema_version: 2
training_dataset_manifest_hash: "<computed>"
training_config_hash: "<computed>"
seed: 42
metrics:
  lap_time_mae_s: "<measured>"
  calibration_error: "<measured>"
status: validated_or_experimental
```

Dünya save dosyası bu manifestteki sabit sürüme referans vermeli; model dosyası silinirse kullanıcıya açık kurtarma mesajı çıkar, sessizce başka modele geçme.

---

## 9. ML MODEL 2: STAGE 1 — PROFILE-CONDITIONED RACE STRATEGIST

### 9.1 İlk soru: Modelin amacı ne?

Bir sürücü için yarış zamanını veya beklenen sportif sonucu en iyi hale getirirken, **o sürücünün profilinden türeyen beceri/tercih ve yasal kısıtlarını** gözetmek. Bir strateji bir sürücüde iyi, başka sürücüde kötü olabilir çünkü sürücünün lastik kullanımı, stabilitesi, risk toleransı farklıdır.

Sadece `minimum_lap_time` optimize etmek yetmez. Yarışı bitirmek, son turda doğru lastikte olmak, pit kaybı, güvenilirlik, ceza ve hava riskleri önemlidir.

### 9.2 Gözlem vektörü `StrategyObservation`

```text
# Current state, only known at decision time
lap_index, laps_remaining, sector_index, gap_to_rival_if_visible,
current_position, current_compound, tyre_age, available_tyres,
estimated_fuel_state, cumulative_race_time, track_temp,
air_temp, rain_intensity, surface_wetness, observed_weather_history,
weather_forecast_distribution, safety_car_state, pit_lane_loss_estimate,
track_embedding, car_performance, driver_profile_vector,
strategy_history, valid_action_mask
```

Stage 1'de `gap_to_rival` yalnızca dış trafik etkisini modellemek için girdi olabilir; rakip karakter tahmini yoktur. Eğer Stage 1 tasarımını saf time-trial ile başlatırsak ilk RL eğitiminde bu alanları mask ile kapat.

### 9.3 Eylem uzayı

En başta ayrık/faktörize:

```python
PIT_ACTION = [STAY_OUT, PIT_SOFT, PIT_MEDIUM, PIT_HARD, PIT_INTER, PIT_WET]
PACE_MODE  = [CONSERVE, BALANCED, PUSH]
```

- Eylemler her tick değil **makul karar noktalarında** alınır: `pre_pit_entry`, `lap_end`, `weather_alert`, `SC_started` gibi.
- Pit mümkün değilse tüm `PIT_*` eylemleri maskelenir.
- Her karar noktasında hem PIT hem PACE değiştirme ilk MVP için karmaşıksa, iki seviyeli hierarchical policy kullan: `pace_policy` per-lap, `pit_policy` at decision points.
- Model kararları eninde sonunda `StrategyAction` olarak normalize edilir ve motor/legal validator'a gider.

### 9.4 Eğitim problemi: MDP/POMDP

- **State:** Dünya motorunun tam durumu (sürücüye kapalı gizli parametreleri de içerir).
- **Observation:** Ajana o anda izin verilen kısıtlı gözlemler (geleceğe sızma yok).
- **Action:** Legal pace/pit önerisi.
- **Transition:** Otoritatif simülatör + weather/tyre/driver dynamics + RNG.
- **Reward:** Yarış sonucu ve risk/stint yönetimi kombinasyonu.
- **Episode:** Tek yarış; sonradan sezon.
- **Partial observability:** Gizli gerçek grip, rakibin planı, gelecek yağmur, rakibin risk tercihi gibi bilgiler tahmin edilir, verilmez.

### 9.5 Ödül tasarımı (örnek, kalibre edilmeli)

```text
R_terminal = - normalized_race_time
           + bonus_for_legal_finish
           + optional_position_or_points_reward (Stage 2)

R_step = - weighted_time_delta
         - avoidable_invalid_action_penalty
         - profile_conditioned_risk_cost
         - abnormal_tyre_or_incident_cost

R_total = R_terminal + sum(discounted R_step)
```

Stage 1 saf time optimization için önce normalize edilmiş toplam yarış süresi ve finish bonusu kullan. Riskin hem çevre olasılıklarında hem reward cezasında iki kez ağır cezalandırılmasından kaçın. `risk_appetite` değiştiğinde **her politika için ortak referans ödülü** ayrıca raporla; sadece profile-specific reward ile başarı kıyaslama.

### 9.6 Baseline'lar (zorunlu)

- `B0 Historical reference`: gerçek yarışta kullanılan strateji; bu **geleceği bilen benchmark** değildir, tarihte olan eylemlerin retrospectif referansıdır.
- `B1 Rule-based conservative`: lastik ömrü, hava ve pit-lane kaybına göre basit eşik kararları.
- `B2 Offline strategy enumerator`: uygulanabilir 0/1/2-stop planlarını varyasyonlarla tarar (hava/stok/yasal kurallar uygun olduğunda).
- `B3 Monte Carlo / MPC strategist`: aynı mevcut gözlemden birkaç geleceği örnekler, eylemleri beklenen sonuç ve risk dağılımına göre sıralar.
- `B4 Learned policy`: behavior cloning (eğitim etiketi varsa) veya MaskablePPO/DQN gibi RL. DQN ve PPO adaydır; algoritma yarış motorunun hızına ve eylem biçimine göre seçilir.

**Başlangıç tercihi:** Baseline B1–B3 sağlam çalışmadan B4'e geçme. Hedefimiz 'RL koyduk' değil, **neden daha iyi olduğunu kanıtlayabilmek**.

### 9.7 RL eğitim döngüsü (pseudocode)

```python
for episode in range(total_episodes):
    driver_profile = sample_profile_from_library_or_curriculum()
    race_scenario = sample_track_weather_tyre_scenario()
    obs, info = env.reset(profile=driver_profile, scenario=race_scenario)

    while not info["terminal"]:
        valid_mask = env.valid_action_mask()
        action = policy.act(obs, action_mask=valid_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        policy.observe_transition_if_training(...)
        if terminated or truncated:
            break

    log_trajectory(profile=driver_profile, scenario=race_scenario,
                   episode_return=info.get("return"), seed=info.get("seed"))
```

Bu **algoritmik şemadır**, doğrudan kullanılacak SB3 API'si değildir. Gerçek Gymnasium wrapper'ında `reset()`, `step()`, `observation_space`, `action_space` sözleşmelerine ve MaskablePPO'nun action mask yönergelerine uy. Özellikle resmi MaskablePPO değerlendirme callback'lerini kullan.

### 9.8 Curriculum learning

1. `C0`: Tek sürücü, kuru pist, sabit hava, basit iki lastik bileşiği, risk yok.
2. `C1`: Çoklu kuru lastikler, lastik yaşı ve profil etkileri.
3. `C2`: Farklı pistler/özellikler, farklı pit kayıpları.
4. `C3`: Yağmur geçişleri, forecast belirsizliği, uygun wet tyre seçimi.
5. `C4`: Rastlantısal olaylar, mekanik hata riski, güvenilirlik.
6. `C5`: Sabit trafik / kısmi rakip varlığı, ama bilinçli rakip ajan modellemesi **Stage 2'ye bırakılır**.

Her curriculum evresinin ayrı benchmark ve best checkpoint'i olur. Yeni senaryoda başarısızlık, önceki başarıyı sessizce ortadan kaldırmamalı.

### 9.9 Profil koşullama ve kontrol testleri

**Architecture A (ilk tercih):** ortak MLP actor/critic, `state_features` + `driver_profile_features` concat. Profil değerleri normalize edilir. `track/compound` kategorileri embedding olabilir.

**Architecture B (sonra):** FiLM veya küçük adapters ile profile-dependent modülasyon.

**Gerekirse:** imitation priors ve KL/behavior regularization. Başlangıçta klasik RL'de öğrenilebilir bir şey olmadan sürücü stilini sadece sabit reward ile sürdürmek, modelin politika girdisini görmezden gelmesine yol açabilir; bu yüzden ablation şart.

Zorunlu testler:

- `test_profile_changes_transition_dynamics`
- `test_policy_observation_contains_profile`
- `test_profile_revision_pinned_to_world`
- `test_driver_policy_not_hardcoded_from_name`
- `test_high_risk_not_always_better`
- `test_same_driver_different_weather_changes_strategy`
- `test_no_future_weather_leakage`
- `test_valid_tyre_stock_and_rules`

### 9.10 Çoklu performans metrikleri

Tek bir iyi yarış modelin iyi olduğu anlamına gelmez. En azından:

- Beklenen total race time, P50/P90 outcome, variance.
- `legal_finish_rate`, incident rate, DNF rate.
- `regret_vs_oracle` (oracle geleceği biliyorsa **yalnızca upper bound**, adil policy rakibi değildir).
- `regret_vs_MPC` ve `beat_rate_vs_rules` eşleştirilmiş seed'lerde.
- Compound usage, pit-stop sayısı, lastik kalan ömür, plan revizyon sayısı.
- Profile-diversity index / risk–reward Pareto grafikleri.
- Seen vs held-out race/weather splits.
- Ortalama ve bootstrap confidence interval; deneme sayısı raporlanmalı.

### 9.11 Eğitim ile canlı dünya arasındaki sınır

- `TrainingService`: çok sayıda **ephemeral cloned scenario** üzerinde hızlı headless simülasyon çalıştırır. Bu eğitim ortamları kullanıcı tarafından kaydedilmiş kanonik dünyaya bağlı değildir.
- `ModelRegistry`: eğitim sonuçları/artefact'lar/version/metrics tutar.
- `WorldSession`: yalnızca **explicitly pinned model_version** kullanır.
- Eğitim bitince kullanıcıya "New trained strategist available" bildirimi. Canlı dünyanın ortasında otomatik weight swap yok.
- `Try new policy in fork` ile kullanıcının yeni modeli ayrı alternatif zaman çizgisinde deneyebilmesi sağlanır.

### 9.12 Sürücüye özgü öğrenme deneyleri

Deney seti:

- **E1:** Policy conditioned on driver profile vs no profile.
- **E2:** Max-style / Hamilton-style / generic profile, same car/same weather/same seed.
- **E3:** Same driver profile, Ferrari-like vs McLaren-like car parameters. Driver skill vs car performance etkilerini ayır.
- **E4:** Learned profile policy vs handcrafted profile-specific heuristics.
- **E5:** Training on dry, test on wet transitions; adaptation başarısı.
- **E6:** Driver profile weights frozen vs modifiable latent variables; kimlik tutarlılığı ve uyarlanma arasındaki trade-off.
- **E7:** Farklı reward tanımlarının risk davranışını bozmadan sonuç değiştirmesi.

Her deney için `config`, training/eval seed'leri, dataset/model/profile hash'leri, sonuç metrikleri ve değerlendirme senaryoları kaydedilmeli.

---

## 10. STAGE 2: RAKİPLER, TAKIM ARKADAŞLARI, TAKTİKLER

**Giriş koşulu:** Stage 1 tek sürücü/tek takım strateji benchmarkları başarıyla raporlandı, replay/save ve model sürümleme güvenilir.

### 10.1 Çoklu ajan arayüzü

Yeni `Actor` tipleri:

- `DriverAgent`: pace mode, attack/defend, tire preservation, risk mode.
- `RaceEngineerAgent`: pit önerisi, hava risk yönetimi.
- `TeamStrategistAgent`: iki sürücüyü birlikte optimize eder.
- `OpponentBeliefModel`: rakiplerin gizli parametreleri hakkında olasılıksal tahmin tutar.

Başlangıçta bütün 20 sürücünün aynı anda farklı RL modelleri olması gerekmiyor. Stage 1 ortak policy + profil girdileri, ilk çoklu ajan mimarisinin temelini oluşturur.

### 10.2 Eylemler ve bilgilerin sınırı

- `attack_level`: normal / aggressive; `defend_level`: normal / cover_inside / preserve_tyres; ilk etapta yüksek seviye stratejik eylemler, direksiyon motoru değil.
- `team_order`: hold_position / swap / pit_priority / split_strategy (yıl kuralları doğrulanmış).
- `opponent_belief`: rakibin pit window ve risk tahmini; **ground-truth gizli policy doğrudan observation'a eklenmez**.
- `communication`: takımın gönderdiği sınırlı structured mesajlar (ör. `target_delta_s` / `pit_this_lap` / `save_tires`).

### 10.3 Takım–sürücü anlaşmazlığını modelleme

Örneğin sürücü kendi yarışını kazanmak için `push` ister, takım ikinci aracı korumak için `conserve` önerir. Çakışma için:

- Etki ve otorite: en son yasal komutu takım stratejisti mi verir, sürücü reddedebilir mi?
- Oyun tasarımı: V2'de takım kararlarını `TeamOrderResolver` ile çöz, `driver_compliance_probability` ilk başta profile + race context; V3'te sosyal `trust` da etkiler.
- Her çakışma event log'a `proposed`, `accepted/rejected`, `reason` ile girer.
- Takım emirleri rastlantısal çatışmalara dönüşmesin; hedefleri açık ve izlenebilir olsun.

### 10.4 Öğrenme süreci

1. Sabit kural tabanlı rakiplere karşı tek öğrenen sürücü.
2. Farklı profil dağılımındaki ajanlar ile turnuva.
3. Öğrenen rakipler ve historical policy pool (`league training`).
4. Takım stratejisti vs iki bağımsız sürücü politikasının karşılaştırması.
5. İletişim/koordinasyon ödül mekanizmalarının ablasyonu.

**Araştırma riskleri:** multi-agent non-stationarity, self-play overfit, reward hacking, equilibrium olmayan davranışlar. Sabit held-out rakip havuzu ve cross-play matrisi şart.

### 10.5 Stage 2 test ve metrikleri

- `same_strategy_all_agents` collapse testi.
- Attack/defend eylemi yasal ve görünür mü?
- Rakip modele future pit sırası sızıyor mu?
- Team objective altında iki sürücünün toplam puanı vs tek-sürücü amaç fonksiyonu.
- Cross-play matrix, exploitability proxy, diversity, relative finish order.
- 20 araçlı simülasyonda race finish invariants; iki aracın aynı pozisyonu atlamaması.

---

## 11. STAGE 3: SOSYAL ZEKA, EKONOMİ, TRANSFERLER VE ALTERNATİF TARİH

Stage 3'te dünya gerçekten **living universe** olur. Bu modülleri `SimulationWorld` durumuna sürümlü olarak ekle, ancak MVP'nin çalışma hızını ve kalitesini tehlikeye atacak erken entegrasyon yapma.

### 11.1 Social state

- `relationship[agent_a, agent_b]`: trust, respect, rivalry, cooperation (ayrı boyutlar, -1..+1 veya 0..1).
- `memory_events`: olay, zaman, katılımcılar, kaynak, duygusal/stratejik önem, unutturma/özetleme politikası.
- `personal_goals`: şampiyonluk, takım liderliği, daha iyi sözleşme, sportif başarı, takım sadakati gibi simülasyon hedefleri.
- `contracts`: takım, başlangıç/bitiş tarihleri, ücret aralığı, çıkış/performans koşulları (kurgusal).
- `team_budget`, `car_development`, `staff`, `fan_support` (tamamı simülasyonla sınırlı).

### 11.2 Social agents nasıl davranmalı?

- Simülasyon motoru önce `WorldSocialState` ve izinli eylem listesini sunar.
- Planner (kural, RL, LLM veya hibrit) bir **structured action proposal** üretir.
- `SocialActionValidator` geçerlilik/para/kontrat/çakışma/otorite kuralını denetler.
- `WorldReducer` geçerli olayı uygular, değişimi event log'a yazar.
- Sosyal ajanın hafızası ve niyeti gözetilir; olay `public/private_in_world` işaretlenebilir.
- Gerçek kişilere yönelik kişisel skandal, suç, özel hayat hakkında kurgu üretme zorunluluğu yok. Daha güvenli ve eğlenceli başlangıç: **sportif ilişkiler, takım güveni, transfer ve finans**.

### 11.3 Sosyal süreçlerin sportif sonuçlara etkisi

Örnekler:

- Sürücü–mühendis güveni, strateji önerisinin kabul olasılığını küçük ölçüde etkiler.
- Takım içi gerilim, takım emri uygulama davranışını etkiler.
- Yeni mühendis/araç geliştirmesi, aracın ayrı `car_performance` parametrelerini etkiler.
- Uzun vadeli transfer, takım kadrosunu değiştirir ama tarihsel yarış snapshot'larını geriye dönük değiştirmez.
- Sosyal müdahalelerin yarış kazandırma gücü sınırlandırılır; lastik ve pace fiziğini keyfî biçimde bypass etmez.

### 11.4 Sosyal LLM'in rolü

LLM **zorunlu altyapı değil**, geliştirilebilir anlatı/planlama katmanıdır:

- Yarış sonrası röportaj önerileri veya takım toplantılarında seçenek üretme.
- Event log'a dayanan doğal dil açıklaması; sonucu değiştirmez.
- Uzun dönem hedefle müzakere taslağı; sonuçları yalnızca validator uygular.
- Önce maliyetsiz structured rule/goal agents, sonra istenirse LLM provider abstraction.
- Dış LLM'e özel kullanıcı verisi, kişisel notlar veya gereksiz bilgi otomatik gönderilmez; kullanıcı opt-in.

### 11.5 Sezon takvimi / ekonomi

Yarışlar arasında sim time ileri yürütülür:

`RACE_FINISHED` → `RESULTS_CONFIRMED` → `TEAM_MEETINGS` → `DEVELOPMENT` → `TRANSFER_WINDOW` → `NEXT_EVENT_PREPARATION`.

Her etkinlik checkpoint oluşturur; sezonlar arası save/load ve branch aynı kurallarla çalışır. Finansal kararlar oyuncunun kontrolüne açılabilir ama oyun dengesi / realism test edilmeli.

### 11.6 Gerçek sürücü isimleri vs kurgusal dünya

Kullanıcının istediği alternatif evren gerçek sürücülerle başlar. UI her sosyal senaryoda `Fictional simulation` notu taşır. Kişilik puanları ve kurgusal sosyal olaylar **gerçek kişiler hakkında doğrulanmış beyanlar değil**, oyun varsayımlarıdır. Kamuya açık kullanımda gerçek markalar/görseller ve gerçek kişilere dair uydurma iç olaylara yönelik ayrıca editoryal değerlendirme yapılır. İstenirse `fictional_names_mode` sunulur.

---
## 12. UYGULAMA MİMARİSİ, TEKNOLOJİLER VE PROJE DİZİNİ

### 12.1 Monorepo yaklaşımı

**Önerilen ilk sürüm:** Python backend/sim + TypeScript/React frontend. Ayrı servisleri mikroservise bölme. Tek `FastAPI` backend süreç içindeki otoritatif `WorldRunner` ve/veya tek background simulation thread/process + SQLite. Ağ bağlantısını frontend ve backend arasında kullan. Araçların yarış durumunu UI mantığında yeniden hesaplama.

- **Backend:** Python 3.11 veya 3.12 ile başla; TUM 3.8 uyumluluk gerektiriyorsa izole adapter/ayrı venv veya fallback kararı ver. Kesin sürümü `M0` sonunda pinle.
- **Dependency/tooling:** `uv` veya pip; `pyproject.toml`, `pytest`, `ruff`, `mypy` (gerekli modüllerde), `pydantic`, `sqlalchemy`/standart sqlite3.
- **ML:** `numpy`, `pandas`, `scikit-learn`, `lightgbm` (opsiyonel), `torch`, `gymnasium`, `stable-baselines3`, `sb3-contrib` (fazı gelince).
- **Data:** `fastf1`, `httpx` veya `requests` OpenF1/Jolpica için, Parquet/Arrow, SQLite.
- **Frontend:** React + TypeScript + Vite; 2D track için PixiJS veya ilk prototipte HTML Canvas/SVG. Harici tile-map/döngülü 3D fizik yok.
- **Real-time:** `GET snapshot` + WebSocket stream (veya ilk adımda polling). **WebSocket state transport'tur, simulation clock değildir.**
- **Testing:** pytest, hypothesis (state invariants), TypeScript unit/component testleri ve Playwright E2E (UI fazında).
- **Model tracking:** ilk fazda dosya manifest + MLflow local (isteğe bağlı); ağır model registry sunucusu şart değil.
- **CI:** GitHub Actions, lint+unit+small simulation integration tests; ücretli data API çağrılarını CI'ın zorunlu parçası yapma.

### 12.2 Önerilen dizin ağacı

```text
f1-alternate-reality/
├── F1_ALTERNATE_REALITY_MASTER_PLAN.md
├── README.md
├── pyproject.toml
├── uv.lock                         # when uv is chosen
├── package.json                    # optional workspace orchestration
├── .gitignore
├── .env.example
├── Makefile                        # optional; Windows alternatives in README
├── configs/
│   ├── app.yaml
│   ├── scenarios/
│   │   ├── monza_2024_historical.yaml
│   │   ├── monza_2024_alternate.yaml
│   │   └── monza_2024_weather_shock.yaml
│   ├── drivers/
│   │   ├── profile_schema.json
│   │   ├── seed_2024_monza_drivers.yaml
│   │   └── archetypes.yaml
│   ├── regulations/
│   │   ├── fia_2024_v1.yaml
│   │   └── fia_2026_v1.yaml        # NOT implemented until verified
│   └── training/
│       ├── pace_baseline.yaml
│       └── stage1_rl_baseline.yaml
├── backend/
│   └── src/f1ar/
│       ├── api/
│       │   ├── main.py
│       │   ├── routes_worlds.py
│       │   ├── routes_races.py
│       │   ├── routes_models.py
│       │   └── websocket.py
│       ├── data/
│       │   ├── fastf1_adapter.py
│       │   ├── openf1_adapter.py
│       │   ├── jolpica_adapter.py
│       │   ├── normalize.py
│       │   ├── validate.py
│       │   └── manifests.py
│       ├── domain/
│       │   ├── identities.py
│       │   ├── driver_profiles.py
│       │   ├── car_state.py
│       │   ├── race_state.py
│       │   ├── team_state.py
│       │   ├── weather.py
│       │   ├── regulations.py
│       │   └── events.py
│       ├── simulation/
│       │   ├── engine_protocol.py
│       │   ├── engine_factory.py
│       │   ├── world_runner.py
│       │   ├── world_reducer.py
│       │   ├── clock.py
│       │   ├── event_scheduler.py
│       │   ├── action_validator.py
│       │   ├── rng_streams.py
│       │   ├── tumsim_adapter.py
│       │   └── lightweight_engine.py
│       ├── policy/
│       │   ├── interfaces.py
│       │   ├── rule_based.py
│       │   ├── monte_carlo.py
│       │   ├── profile_conditioned.py
│       │   └── model_registry.py
│       ├── training/
│       │   ├── pace_dataset.py
│       │   ├── pace_train.py
│       │   ├── gym_env.py
│       │   ├── stage1_train.py
│       │   └── evaluate.py
│       ├── persistence/
│       │   ├── sqlite_store.py
│       │   ├── snapshot_codec.py
│       │   ├── migrations.py
│       │   └── timeline_repository.py
│       ├── championship/          # stage 2+
│       └── social/                # stage 3+
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── app/
│       ├── api/
│       ├── components/
│       │   ├── TrackCanvas.tsx
│       │   ├── RaceControls.tsx
│       │   ├── TimingTower.tsx
│       │   ├── TyreStrategyPanel.tsx
│       │   ├── DriverProfileCard.tsx
│       │   ├── SaveTimelineDialog.tsx
│       │   └── ScenarioComparison.tsx
│       ├── pages/
│       │   ├── WorldLibrary.tsx
│       │   ├── RaceSimulator.tsx
│       │   ├── DriverProfiles.tsx
│       │   └── ExperimentLab.tsx
│       └── tests/
├── data/
│   ├── raw/                        # .gitignore
│   ├── processed/                  # .gitignore unless redistributable
│   ├── manifests/                  # small tracked metadata
│   └── sample_synthetic/           # tracked fixtures, license-safe
├── artifacts/
│   ├── models/                     # mostly .gitignore
│   ├── evaluations/
│   └── screenshots/
├── saves/                          # .gitignore user worlds
├── scripts/
│   ├── bootstrap.py
│   ├── fetch_monza_2024.py
│   ├── build_2024_profiles.py
│   ├── validate_dataset.py
│   ├── demo_headless.py
│   ├── benchmark_engine.py
│   └── inspect_save.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── property/
│   ├── replay/
│   ├── ml/
│   └── fixtures/
├── docs/
│   ├── DECISIONS.md
│   ├── PROGRESS.md
│   ├── KNOWN_LIMITATIONS.md
│   ├── EXPERIMENTS.md
│   ├── ASSUMPTIONS.md
│   ├── LEGAL_AND_LICENSES.md
│   ├── API_CONTRACTS.md
│   ├── DATA_DICTIONARY.md
│   └── GETTING_STARTED.md
└── .github/workflows/ci.yml
```

**Uygulama sırası:** Dizini bir anda boş placeholder dosyalarla doldurma. İlgili milestone geldiğinde gerçek kodla dosya oluştur. `/social` veya `/championship` klasörleri Stage 1'de boş olmak zorunda bile değil.

### 12.3 Domain nesneleri ve kimlikler

- `EventID`: `season/round/session` kimliği ile `world_timeline_event` kimliğini ayır.
- `DriverID`: season-specific entry ve devamlı kişisel identity ayrı; 2024 driver number tek başına benzersiz değil.
- `TeamID`: gerçek sezon snapshot ile alternatif takım nesneleri ayrı; isim değişimleri desteklenir.
- `TrackID` + `TrackLayoutVersion`: Monza'nın geometrisi ve sezon-özgü değişiklikler için.
- `RegulationsVersion`: `fia_2024_v1`, başka sezonlar farklı.
- `DatasetManifestID`, `PolicyVersion`, `DriverProfileRevision`, `EngineVersion` birlikte state provenance yaratır.
- `WorldID`, `TimelineID`, `SnapshotID`, `EventID`: yerel persistent UUID/ULID türü benzersiz ID.

### 12.4 `WorldRunner` kontrol döngüsü (pseudocode)

```python
async def run_session(world_id):
    while session_is_open(world_id):
        commands = await command_queue.drain_in_order(world_id)
        apply_control_commands(commands)  # pause/resume/speed/save/fork at safe boundaries

        if world.status == "PAUSED":
            await wait_for_next_command()
            continue

        if world.status != "RUNNING":
            await wait_for_next_command()
            continue

        # One bounded canonical step; do not run a blocking full race here.
        world, events = engine.step(max_sim_delta_seconds=config.sim_tick_seconds)
        persistence.append_events_and_maybe_snapshot(world, events)
        publish_immutable_view(world, events)
        await yield_control_for_commands()
```

Gerçek implementasyonda yarış physics step CPU-bound ise event loop bloke edilmemeli; tek worker thread/process veya async dışı runner kullan. `apply_control_commands` komutlarını simulation state tutarlılığı ile sırala. `save` isteğine kullanıcıya başarı bildirimi yalnızca transaction commit bittikten sonra dönsün.

### 12.5 Serializasyon ve RNG ayrıntıları

- Uygulama state'i Pydantic/dataclass -> deterministik JSON veya MessagePack (tip sözleşmesi). Float rounding kararı ve codec sürümü sabit.
- `random.Random.getstate()` ve NumPy `bit_generator.state` JSON-compatible kodlayıcıyla kaydedilebilir. PyTorch RNG state byte dizisi güvenli codec/base64 ile taşınır; CUDA nondeterminism ayrı belgelenir.
- ML model ağırlıkları ayrı imzalı/hash'li model artifact'ta; dünya snapshot'ına komple kopyalanmaz. Mümkünse `safetensors`; doğrulanmamış pickle/pth dosyalarını uzaktan yükleme.
- Python/package/engine sürümleri `environment_manifest` ile saklanır; eski world versiyonuna uygun decoder/compatibility check gerekir.
- `state_hash`: debug ve determinism regresyonu için normalize edilmiş canonical state hash'i.

---

## 13. PERSISTENCE VE API TASARIMI

### 13.1 SQLite önerilen tablolar

```text
worlds(id, name, created_at, active_timeline_id, created_from_race_id, status)
timelines(id, world_id, parent_timeline_id, fork_snapshot_id, name, created_at)
snapshots(id, timeline_id, tick, race_sim_seconds, schema_version,
          engine_version, dataset_hash, profile_hash, policy_manifest_json,
          payload_location_or_blob, payload_sha256, created_at, is_consistent)
events(id, timeline_id, sequence, tick, event_type, payload_json, actor_id,
       created_sim_time, created_wall_time)
commands(id, timeline_id, idempotency_key, requested_tick, applied_tick,
         actor_id, command_type, payload_json, result_json)
model_versions(id, model_name, artifact_path, artifact_sha256,
               feature_schema_version, metrics_json, created_at, status)
profile_revisions(id, driver_id, season_id, schema_version, payload_json,
                  payload_sha256, origin_label)
dataset_manifests(id, event_id, payload_json, sha256)
experiment_runs(id, kind, config_hash, dataset_hash, profile_hash,
                model_hash, seed, metrics_json, artifact_dir)
```

`profiles`, `rosters`, `cars`, `teams` ayrı boyut tablolarına da ayrılabilir. Snapshot payload tüm world state'i içerdiği için DB ilişkileri karmaşıklaştırmadan incremental başla.

### 13.2 Temel API sözleşmeleri (ilk sürüm)

| Verb | Endpoint | Sonuç |
|---|---|---|
| `GET` | `/api/health` | Backend/DB ve engine durumu |
| `GET` | `/api/worlds` | Kayıtlı dünyalar, son save, aktif timeline |
| `POST` | `/api/worlds` | Yeni world; mode, race, seed, profile revision |
| `GET` | `/api/worlds/{id}` | Current canonical world snapshot / metadata |
| `POST` | `/api/worlds/{id}/start` | RUNNING |
| `POST` | `/api/worlds/{id}/pause` | PAUSED ack with applied_tick |
| `POST` | `/api/worlds/{id}/resume` | RUNNING ack |
| `POST` | `/api/worlds/{id}/speed` | x1/x2/x5/x10/x25 |
| `POST` | `/api/worlds/{id}/step` | PAUSED iken 1 tick/sector/lap |
| `POST` | `/api/worlds/{id}/save` | Snapshot metadata; commit sonrası success |
| `POST` | `/api/worlds/{id}/fork` | Branch timeline ID; optional decision override |
| `POST` | `/api/worlds/{id}/load` | Belirtilen save'i canlı oturuma yükler |
| `GET` | `/api/worlds/{id}/timelines` | Branch ve save geçmişi |
| `POST` | `/api/worlds/{id}/actions` | Pit/pace/user override, idempotent |
| `GET` | `/api/races/{race_id}/reference` | Historical reference info |
| `GET` | `/api/drivers/{driver_id}/profile` | Profil ve source tags |
| `GET` | `/api/models` | Pinned model versiyonları/metrics |
| `WS` | `/ws/worlds/{id}` | State snapshots/deltas/events, authoritative sequence |

`POST /fork` ve `/load` parametrelerini `docs/API_CONTRACTS.md` içinde açıkça sabitle. `load` üzerine yazmak yerine yeni timeline/session ID ile çalışmayı tercih edebilirsin, ama kullanıcı UI davranışı net olmalı. Bir dünya `RUNNING` iken `fork` ancak güvenli canonical snapshot'tan yapılır.

### 13.3 Örnek API payloadları

```json
{
  "action_type": "PIT_REQUEST",
  "actor_id": "team_mclaren_2024",
  "driver_id": "lando_norris_2024",
  "requested_compound": "HARD",
  "requested_at_tick": 615,
  "idempotency_key": "uuid-from-client"
}
```

```json
{
  "name": "Monza 2024 — Norris pits on lap 18",
  "source_snapshot_id": "snapshot_615",
  "seed_mode": "COMMON_EXOGENOUS_RANDOMNESS",
  "policy_overrides": {},
  "action_overrides": [{"driver_id": "lando_norris_2024", "type": "PIT_REQUEST", "compound": "HARD"}]
}
```

### 13.4 Kullanıcı ve client state

- `Play/Pause/Save` butonları ancak server ACK gelince başarılı gösterilir; optimistic UI state **yalnızca pending** olarak belirir.
- Son WebSocket sequence ID veya tick daha eskiyse state delta yok sayılır/yeniden snapshot istenir.
- Bağlantı koparsa simülasyonu isteğe göre server'da pause etmek veya local-session closed politikasına göre durdurmak güvenli default; **MVP için bağlantı kesilince auto-pause** seç. İsteğe bağlı background unattended simulations gelecek özellik.
- Kaydedilmiş dünyalar diskte kalır; frontend LocalStorage otoritatif state değildir.

---

## 14. FRONTEND: 2D YAŞAYAN YARIŞ DENEYİMİ

### 14.1 Ana ekranlar

**World Library (ilk açılış):** `Create World`, `Load`, `Continue`, kayıtlı evren kartları, mevcut sim time, Monza/event, son save, engine/model/profil sürümü, alternatif timeline sayısı.

**Race Simulator (ana kullanım):**

```text
┌───────────────────────────────────────────────────────────────────┐
│ F1 ALTERNATE REALITY  |  MONZA 2024  | LAP 18/53 | ☀ 24°C | PAUSED│
├──────────────────────────────────────┬────────────────────────────┤
│                                      │  TIMING TOWER              │
│            2D TRACK VIEW             │  P1 ... DRIVER / GAP       │
│       cars move on real geometry     │  P2 ...                    │
│       highlighted car + pit lane     │  P3 ...                    │
│                                      │  TYRE / STINT              │
├──────────────────────────────────────┤                            │
│ PLAY PAUSE STEP  SPEED   SAVE  FORK  │  DRIVER PROFILE            │
├──────────────────────────────────────┤                            │
│ ACTIVE DRIVER  | PACE | TYRE | PIT   │  AI DECISION EXPLANATION   │
│ CURRENT PLAN   | WEATHER FORECAST    │  Alternatives / Confidence │
├──────────────────────────────────────┴────────────────────────────┤
│ TIMELINE: [events, pit stops, SC, weather, overrides, snapshots] │
└───────────────────────────────────────────────────────────────────┘
```

**Driver Profile editor:** Sürücü kimliği; verified historical facts vs editorial archetype ayrımı; slider'lar; `profile_version`; `fork world to apply changes`. Model stage/behavior explainability paneli.

**Race Analysis:** Historical vs actual simulated vs alternative branch; lap/pit charts, compound/stint bar, position changes, percentiles/uncertainty.

**Experiments (daha sonra):** Profil ablasyonları, model versiyonları, benchmark reports ve training log.

**Championship Map (Stage 2+):** Takvim, puan, bütçe, transfer ve driver relationships.

### 14.2 Kontroller

`Play`, `Pause`, `Save`, `Save As`, `Load`, `Fork`, `Resume`, `Step Tick`, `Step Segment`, `Step Lap`, `Speed`, `Take Control`, `Release Control`, `Set Pit Strategy`, `Select Driver`, `Compare Timelines`, `Edit Driver Profile (fork)`.

Özellikle kullanıcı hikâyesi: “Yarışı durdur → mevcut kaydı sakla → kendi takımımı kontrol et → farklı karar ver → sonuçları gör → ana hikâyeye dön.” Bunu ilk sürümde en az tek driver/pit eylemiyle doğrula.

### 14.3 Strateji panelinin içeriği

- Şu anki lastik / mevcut stint turu / tahmini kalan performans.
- Son turlar ve forecast aralığı.
- AI önerisi `STAY OUT` veya `PIT FOR HARD`, geçerlilik durumu.
- `Predicted outcome if action A/B` ayrık simülasyonlar üzerinden dağılım.
- Explanation verisi: `top factors` ve counterfactual sensivity (uydurma LLM açıklaması yerine feature attribution/solver outputs).
- UI “model predicted” / “assumed” / “historical observed” ayrımını açıkça yapmalı.

### 14.4 Görsel tutarlılık

- 2D pist geometry gerçek ölçülerden normalize edilir; seyirci ekranı responsive.
- Kaydedilen araç ilerleme `distance_along_track_m`, görüntü koordinatı interpolasyon.
- `paused` iken animasyon durur, render aynı yerde kalır.
- `Replay mode` için historical position interpolation; `Alternate mode` için simüle edilmiş progress interpolation.
- Kurgusal pit lane görseli varsa `schematic / approximate` etiketi.
- Önce basit renkli araba işaretleri; marka logo lisansları doğrulanmadan logo paketleri otomatik indirme.

### 14.5 Erişilebilirlik / performans

- Klavye: space = play/pause, `S` = save (form odaklı durumlarda engelle), sol/sağ = paused stepping.
- Renk körlüğü dostu lastik sınıf etiketleri; yalnızca renk kullanma.
- Rendering 30–60 FPS hedef, canonical state tick'i backend otoritesinde.
- 20 araç animasyonu makul yerel laptop'ta çalışmalı; animation bozuk olsa bile backend/save çalışmaya devam etmeli.

---

## 15. İLK MVP'NİN KESİN KAPSAMI VE DIŞARIYA ATILANLAR

### 15.1 Minimum oynanabilir dikey dilim (Vertical Slice A)

İlk anlamlı demo şu kullanıcı akışını **uçtan uca gerçek backend** üzerinde gerçekleştirmeli:

1. `2024 Monza` historical datayı lokal cache'den aç.
2. 2D Monza pistini ve en az birkaç aracı göster (başlangıçta **20 araç hedef**, veri problemi varsa bu eksikliği dürüstçe belirt).
3. `Play` ile simülasyon ilerlesin; araçların canonical progress'i değişsin.
4. `Pause` anında sabitlensin.
5. `Save` ile world diske yazılsın.
6. Uygulamayı tamamen yeniden başlat → `Load` → aynı yarış zamanı ve sürücü/lastik/pist durumu gelsin.
7. `Fork` → bir sürücünün pit planını değiştir → branş sonucu kaynak timeline'ı bozmadan ilerlesin.
8. En az kural tabanlı profil koşullu bir agent öneri üretsin; önerinin nedenleri kaydedilsin.

**Bu dilim hazır olmadan** RL training dashboard, LLM röportajları, ekonomiye geçme.

### 15.2 Stage 1 MVP (Vertical Slice B)

- Gerçek başlangıç kadrosu ve 2024 ruleset.
- Güvenilir pit/tyre/traffic/weather proxy simülasyonu.
- En az 2 profile-conditioned drivers + tüm sürücüler için düzenlenebilir başlangıç profilleri.
- Profile-conditioned pace/degradation model veya anlamlı baseline.
- Profile-conditioned race strategist (en az baseline, sonra öğrenilmiş model).
- 100+ paired-run karşılaştırmaları ve belirsizlik raporu (hesaplama maliyetine göre adaptif).
- Kullanıcı müdahalesi sonrası sonuç analizi.
- Save/resume/branch determinism regression suite.

### 15.3 Daha sonra yapılacaklar (not MVP)

- 3D sürüş fiziği.
- Gelişmiş multi-agent overtaking/defending.
- Sözleşme ve transfer motoru.
- LLM ile doğal dil pazarlık.
- Tüm sezon canlı ekonomi.
- Cloud hosting/multi-user/multiplayer.
- Takım logoları, resmi yayın sesi, sponsor grafikleri.

---
## 16. GELİŞTİRME YOL HARİTASI — GERÇEKÇİ AŞAMALAR VE ÇIKIŞ KRİTERLERİ

> Süreler **hedef aralıklarıdır, garanti değildir**. Tek geliştirici ve 6–10 saat/hafta için ilk yerel oynanabilir prototip yaklaşık 6–10 hafta, kapsamlı Stage 1 (öğrenen stratejist, testler, interaktif simülasyon) yaklaşık 4–7 ay, Stage 2 ve Stage 3 ise modüler şekilde sonrasında ilerler. Eski TUM adapter, data licensing, UI ve ML validasyonu için takvim gerektiğinde güncellenir. **Kalite kapılarını atlayarak tarih yetiştirme.**

### M0 — Feasibility, environment, source audit (1–2 hafta)

**Amaç:** Geliştirme risklerini erkenden kaldırmak ve güvenilir başlangıç oluşturmak.

**Görevler**

- [ ] Repository ve Python/TS minimum araçlarını oluştur; Python sürümünü pinle.
- [ ] Git/Gitignore, README, `docs/DECISIONS.md`, `docs/PROGRESS.md`, `docs/KNOWN_LIMITATIONS.md`, test runner.
- [ ] FastF1 cache ile Monza 2024 `session.load()` dene; round/session tanımı doğrula.
- [ ] En az laps/stints/weather/positions verisi için completeness raporu üret.
- [ ] OpenF1/Jolpica rate limit, kullanılabilir endpoint, source/license notlarını doğrula.
- [ ] TUM Race Simulation'ı **izole sandbox** içinde örnek eski yarışı çalıştır; dependency ve performans raporu oluştur.
- [ ] TUM `pause`/`step`/`serialize` için ihtiyaç duyulan değişiklikleri yaz. Uygun değilse explicit fallback A/B/C kararı ver.
- [ ] **Profiles FIRST:** `DriverProfile`, `DriverArchetype`, `ProfileRevision` veri şeması, seed ve validator yaz.
- [ ] Gerçek Monza 2024 driver/roster ID eşlemesini oluştur; profil verilerini editorial assumption olarak etiketle.
- [ ] `WorldState` ve `WorldRunner` için minimal sentetik 2-driver demo, deterministic step/pause/serialize/load/fork.
- [ ] `pytest` ile temel state ve profil testleri.

**Çıktılar:** `data quality report`, `simulator adapter decision memo`, örnek driver seed dosyası, çalışan headless save/load/fork test.

**M0 kabul kapısı**

```text
[PASS] python -m pytest tests/unit tests/replay
[PASS] python scripts/demo_headless.py --seed 42 --pause --save --reload --resume
[PASS] A repeat run produces identical canonical state hash in deterministic mode
[PASS] DriverProfile fields validate and are included in observation schema
[PASS] Simulator decision, license notes, and data limitations documented
```

**Claude Code, burada dur:** TUM çalışmazsa başarısız diye tüm projeyi durdurma; fallback seçimini gerekçelendir, çalışan minimal headless flow bırak.

### M1 — Veriyi sabitle, Monza 2024 historikal katmanı (2–3 hafta)

**Görevler**

- [ ] `scripts/fetch_monza_2024.py` repeatable, cache/retry/rate-limit.
- [ ] Typed raw/processed schema, UTC/units, lap/tyre/pit validation.
- [ ] Dataset manifest SHA256, DVC şart değil.
- [ ] 2024 Monza grid, race results, pit/stint changes verisini doğrula.
- [ ] TUM pist merkez çizgisini import et; race geometry scale ve 2D coordinates doğrula.
- [ ] Historical replay engine ile tura göre leaderboard; reference event list.
- [ ] Synthetic sample fixture ekle ki testler internet olmadan çalışsın.
- [ ] `historical_replay` ve `counterfactual` modlarını kod seviyesinde açıkça ayır.

**M1 DoD:** Bir CLI komutu gerçek Monza'yı indirip doğrulayabilmeli, sonraki komut cache'den veri yüklemeli; sonuç veri tablosunda 20 araç/53 tur tutarlılık kontrol edilmeli (retirement/lapped cars gibi istisnalara doğru yaklaş). Kaynağın data kapsamındaki eksiklikler raporlanmış olmalı.

### M2 — Canonical Race Engine + persistent world (2–4 hafta)

**Görevler**

- [ ] Engine adapter contract ve `step(max_delta)`.
- [ ] Lastik, tyre age, pit transitions, weather event ve grip proxy durumları.
- [ ] Driver profile → pace/tyre/weather effect (önce parametrik testable baseline).
- [ ] `WorldRunner`, command queue, `RUNNING/PAUSED` state machine.
- [ ] SQLite snapshot+events+commands; idempotency + schema version + checksum.
- [ ] Named saves ve autosave.
- [ ] Fork timelining ve common-exogenous-randomness test.
- [ ] İn-memory state'ten tam yeni süreçte restore testi.
- [ ] Her sim tick'te state invariants; `profile_revision`/`model_hash` pinleme.
- [ ] Batch headless benchmarking (100 episodes/day if feasible).

**M2 DoD:** `pytest` replay/property/integration paketleri pass. Pause sonrası hiçbir canonical alan ilerlemiyor. Fork orijinali değiştirmiyor. Bir save'den 1000 tick sonra state hash karşılaştırması yeşil. M0'daki sınırlı placeholder yerine gerçek state içeriği saklanıyor.

### M3 — İlk görsel oynanabilir 2D yarış (2–3 hafta)

**Görevler**

- [ ] FastAPI GET/POST endpoints, data contracts ve WebSocket state stream.
- [ ] React/Vite app bootstrap, TrackCanvas / TimingTower / RaceControls.
- [ ] Monza centerline + 20 car sprites, race progress, UI event markers.
- [ ] `pause/save/load/fork` için gerçek backend bağlantısı.
- [ ] Kullanıcı bir araç/sürücü seçebilmeli, lastiği ve profili görebilmeli.
- [ ] Profillerin `editorial_assumption` etiketleri gösterilsin.
- [ ] Race mode indicator: `historical replay` / `alternate simulation`.
- [ ] UI'da `x1`, `x5`, `x10`; `save success` yalnızca backend ack ile.
- [ ] E2E: play → pause → save → reload browser/backend → resume → fork.

**M3 DoD:** 2D demo gerçekten kullanıcı girdisiyle çalışıyor ve tam uygulama yeniden açılıp kaldığı yerden devam edebiliyor. İstersen bunu portföy için kısa GIF/video ile belgeleyebilirsin; üretim kodu ondan bağımsız.

### M4 — Yarış modelleri ve kalibrasyon (3–5 hafta)

**Görevler**

- [ ] `pace predictor` dataset ve leakage audits.
- [ ] Profile/car/track/tyre/weather feature schema.
- [ ] Baseline rolling median / linear / LightGBM.
- [ ] Sonra profile-conditioned MLP (gerekli iyileşme görürsek).
- [ ] MAE ve confidence interval, per-track/per-driver sonuçlar.
- [ ] Pit-stop, weather, tyre curves ve traffic model kalibrasyonu.
- [ ] 2024 Monza'nın historical reference verisine karşı “calibration” raporu.
- [ ] Feature importance / ablation, model registry manifest.
- [ ] Model açık değilse deterministic baseline ile işlev devam etmeli.

**M4 DoD:** Test yarışlarında baseline raporu, profile-conditioned model ablasyonu, tüm model sürümlerinin track edilmesi ve sim engine'e güvenli entegre edilmiş predictor. “Gerçek veriyle eğitilmiş” deniyorsa dataset/feature train split manifesti var.

### M5 — Stage 1 Strateji öğrenmesi (4–7 hafta)

**Görevler**

- [ ] Profil-conditioned rule strategist ve legal masks.
- [ ] Offline strategy enumerator ve MC/MPC baseline.
- [ ] Gymnasium `RaceStrategyEnv` (headless fast simulation).
- [ ] Curriculum C0-C3, hız doğrulaması.
- [ ] MaskablePPO veya DQN'i deterministik config ile eğit.
- [ ] Learned policy checkpoint/registry; evaluation separate seeds.
- [ ] Ablations: profile/no-profile; weather/no-weather; rules/RL.
- [ ] UI `AI suggestion` ve candidate strategies / outcome distribution.
- [ ] `Fork and try model` akışı: model upgrade ana timeline'a sessizce geçmez.

**M5 DoD:** Eğitilmiş agent bütün test senaryolarında baselinelere karşı **dürüstçe** raporlanıyor. Başarısızsa da sonuç ve teşhis var; yalnızca `training finished` demek yetmez. Profile-conditioned behavior farkları ölçülüyor. Bir yarışı yeniden açınca aynı model sürümü kullanılıyor.

### M6 — Counterfactual Analysis Lab (2–4 hafta)

**Görevler**

- [ ] Historical branch point seç (ör. Monza Lap 18).
- [ ] Her strateji için eşleştirilmiş seed weather/incident scenarios.
- [ ] Branch policy, opponent baseline, pit plan değişiklikleri.
- [ ] `What-if` probability distribution (mean, p10, p50, p90, win_prob under sim assumptions).
- [ ] Simulated outcomes vs historical reference vs oracle best-case ayrımı.
- [ ] `Replay`, `Compare`, `Restore`, `Fork Again` UI.
- [ ] Export experiment manifest + figures + table.
- [ ] En az 3 değişik alt senaryoda demonstrasyon: kuru, hava geçişi, traffic/SC.

**M6 DoD:** Kullanıcı kaydettiği yarıştan en az iki farklı strateji dallandırıp iki ayrı sonuç dağılımını karşılaştırabilir; ana timeline etkilenmez.

### M7 — Stage 1 release candidate / technical debt (2–3 hafta)

- [ ] Reproducible seed, save schema migration, profiling, README tutorial.
- [ ] Dependency audit ve network/secret safety; offline regression.
- [ ] Data licenses/attributions, legal notes, disclaimers.
- [ ] 10 dakikalık demo ve teknik README (limitations dahil).
- [ ] Unit/integration/ML/replay/E2E CI yeşil.
- [ ] Driver profile editor ve manual override UI stabil.

**Bu noktada Stage 1 tamamlanmış sayılabilir.** Stage 1 toplam takvimini M0–M7 iş yükü ve gerçek ilerleme hızı belirler. Bunları eksiksiz 12 haftaya sıkıştırmak şart değildir.

### M8 — Stage 2 temel rekabet (2–3 ay; birden çok küçük release)

1. Rakip belief model baseline.
2. High-level attack/defend/traffic events.
3. Team strategist 2-driver optimization.
4. Team orders / communication.
5. Self-play veya league training.
6. Opponent generalization evaluation.
7. Sezon puan sisteminin versiyonlu uygulanması.

**DoD:** Çoklu ajanlar birbirlerine göre davranış değiştiriyor; profiling, save/replay ve 20-car race invariants yeşil. Rakip davranışına dair açıklamalar gözlemlenebilir olaylara dayanıyor.

### M9 — Championship continuation & evolving identities (2–3 ay)

1. Tüm sezon takvimi ve tarihsel kadro snapshot'ları.
2. Dünya takvimi, yarışlar arasında state persistence.
3. Pilotun learned policy/profilinin bir yarıştan diğerine taşınması.
4. Takım/araç geliştirme ve finansın basit kuralları.
5. Player `take/release control`.
6. Sezonlar arasında branch/timeline continuation.

**DoD:** İlk Grand Prix sonrası dünya kapanıp açıldığında puanlar, takvim, profil/adaptasyon ve bütçe/sürücü değişimleri korunur.

### M10 — Stage 3 social intelligence (uzun vadeli iteratif)

1. `SocialState` veri modeli ve relation event log.
2. Deterministic goal-based social agents (LLM değil).
3. Sözleşme/transfer ve takım içi güven.
4. Sosyal eylemlerin validasyonu ve kontrollü spor etkisi.
5. İçerik/anlatı layer'ı (isteğe bağlı LLM).
6. Social policy training/learning experiments.
7. Social narrative UI, timeline ve branch.

**DoD:** Oyuncu önceki sezonun sürücü gerilimini, transferini, yeni takım uyumunu açıp inceleyebilir; neden-sonuç zinciri event kayıtlarında bulunur; dünya save/load bunu birebir korur.

### 16.1 Haftalık çalışma ritmi (6–10 saat)

- ~1 saat: bu haftanın milestone/gate, kaynak ve riskleri gözden geçir.
- ~4–6 saat: bir dikey feature'ın implementasyonu.
- ~1–2 saat: test, bug, benchmark.
- ~1 saat: `docs/PROGRESS.md` ve kısa demo çıktısı.

**Her hafta çalışan mevcut sürüm** korunur. Bir haftalık görev, kullanılabilir demo ortaya çıkarmıyorsa en azından CLI test ile doğrulanabilir olmalı.

---

## 17. USER STORIES, EDGE CASES VE KABUL KRİTERLERİ

### US-01 — Yeni dünya

**Given** tarihsel Monza 2024 datası indirilmiş. **When** kullanıcı `Create World` seçer. **Then** seçilen start roster/driver profiles/ruleset/model version ve seed ile yeni `PAUSED` dünya oluşur. Aynı event seçilse bile `WorldID` yenidir.

### US-02 — Play/Pause

**Given** dünya `RUNNING`. **When** `Pause` komutu gelir. **Then** canonical tick üzerinde ack döner, session state `PAUSED` olur, sim clock ilerlemez. UI'da geriye doğru ani sıçrama veya geçersiz tur bilgisi görülmez.

### US-03 — Save and return later

**Given** kullanıcı 31. tur sırasında paused. **When** `Save` → backend kapanır → 2 gün sonra `Load`. **Then** aynı state (arabalar, lastikler, weather, sürücüler, pit queue, profile/policy versions, RNG) geri gelir. Kaydetme zamanının gerçek dünyadaki tarihi, yarış sim-time'ını ilerletmez.

### US-04 — Fork timeline

**Given** world A Lap 18 save. **When** world B fork edilir ve Piastri'nin pit seçimi değiştirilir. **Then** B'nin yarış geleceği yeniden simüle edilir, A'da hiçbir veri değişmez. İki world aynı referans geçmişi paylaşabilir.

### US-05 — Driver-specific Stage 1 learning

**Given** driver profile'lar training başlamadan yüklü. **When** politika farklı profillerle eğitilir ve aynı koşulda değerlendirilir. **Then** öğrenme manifesti profilleri içerir; profil manipülasyonuna karşı farklı davranış eğilimleri ölçülebilir; her sonuçla beraber toplam yarış performansı raporlanır.

### US-06 — Weather adaptation

**Given** forecast başlangıçta kuru, sonraki turlarda yağmur olasılığı artıyor. **When** stratejist karar verir. **Then** sadece o andaki gözlemler/forecast ile pit planı önerir; gerçek geleceği bilmez; wet/inter legal constraints kontrol edilir.

### US-07 — Model upgrade

**Given** dünya policy_v1 ile kaydedilmiş. **When** policy_v2 eğitilir. **Then** saved world v1 ile restore edilir; v2 seçmek isteyen kullanıcı `fork with policy v2` seçer.

### US-08 — User intervention

**Given** AI `STAY OUT` önerdi. **When** oyuncu `PIT FOR HARD` seçti. **Then** karar legal ise komut gerçekleşir, actor `USER` log'a yazılır, sonraki dünya olayları yeni stratejiden etkilenir.

### US-09 — Save failure

**Given** disk dolu veya checksum hatası. **When** Save gerçekleşmez. **Then** UI yanlışlıkla success göstermez; son sağlam checkpoint bozulmaz, okunabilir hata verilir.

### US-10 — Partial data availability

**Given** bazı hava veya telemetry verileri eksik. **When** dataset import edilir. **Then** `missingness_notes` üretilir; güvenilir forecast üretilemiyorsa fallback prior kullanılır ve bu bilgi UI/log'da görünür.

### US-11 — Stable Championship continuation

**Given** sosyal ilişkiler/puanlar/kontratlar mevcut. **When** yeni yarışa geçilip pause/save/load yapılır. **Then** geçmiş ve yeni dünya state'i tutarlı şekilde korunur.

### US-12 — Profile evolution vs identity

**Given** sürücü tarzı profili `v1`, learned policy `v3`. **When** takım transferi olur. **Then** driver identity aynı kalır; team/car değişir; learned policy kontrollü devam eder; geçmiş timeline profil/araç versiyonları sabit kalır.

---

## 18. SİSTEM TEST STRATEJİSİ

### 18.1 Unit tests

- Driver profile range/schema defaults / source labels.
- State transition legality, tyre inventory, pit phase transitions.
- `lap_number` ve `tyre_age` off-by-one önlemleri.
- `RegulationSet` yıl bazlı kontroller.
- `PacePredictor` negatif/NaN handling, missing feature masks.
- Weather/track factor yön testleri (ör. ıslak pist + uygunsuz slick lastik makul cezaya yol açmalı; mutlak model formunu garanti etme).

### 18.2 Property/invariant tests

Her tick'te aşağıdakiler sınanabilir:

```text
world.tick >= 0
world.sim_clock >= 0
0 <= car.distance_along_track_m < track.length_m  (track length > 0)
0 <= car.lap <= race.total_laps
car.tyre_age >= 0
car.fuel_estimate >= 0 OR explicit out_of_fuel_state
unique driver IDs in race starter list
legal tyre compound / regulatory constraints
only one authoritative transition per command sequence
snapshot_hash matches encoded snapshot
no change in paused state
```

**Sıra ve pozisyon:** DNF/lapped cars dahil race ordering için explicit scorer gerekir. Sadece `distance_along_track_m` üzerinden pozisyon hesaplama yetersiz olabilir; `total_distance_progress_m`, yarış statüsü ve tamamlanan turlar dikkate alınmalı.

### 18.3 Integration tests

- FastF1 -> normalized Parquet -> `RaceConfig` -> world initialization.
- Engine -> snapshot store -> backend restart -> resume.
- Engine -> policy interface -> action validator -> event log.
- API -> DB -> websocket -> UI (E2E).
- Replay + two branches -> comparable experiment output.

### 18.4 Deterministic simulation test

```python
# Pseudocode only
world_a = create_world(seed=123, policy="fixed_policy_v1")
advance(world_a, ticks=1000)

world_b = create_world(seed=123, policy="fixed_policy_v1")
advance(world_b, ticks=400)
snap = save(world_b)
world_b_restarted = load_in_fresh_process(snap)
advance(world_b_restarted, ticks=600)

assert canonical_hash(world_a) == canonical_hash(world_b_restarted)
```

Yalnızca Python objesini deepcopy edip karşılaştırmak yeterli değil; başka süreçte data/model version ve RNG yüklenmiş olmalı.

### 18.5 ML tests

- Leakage testleri (`available_at_tick <= decision_tick`).
- Train/test event separation.
- Reward sign sanity: daha yavaş tamamlanan aynı koşulda daha yüksek reward olmamalı (profile penalty dahil araştırma açıklaması).
- Seed paired evaluation.
- Action mask/invalid action share.
- Pace calibration, profile sensitivity, no name-only shortcut.
- Failure case catalog (wet shock, DNF, SC, low tyre inventory, pit confusion).

### 18.6 CI / dev hız hedefleri

- Fast unit test paketi tercihen <2 dakika.
- Integration/mini sim test paketi tercihen <5 dakika.
- Full Monte Carlo/ML benchmark gece veya isteğe bağlı ayrı komut; her commit'te saatler süren eğitim koşma.
- Network gerektiren data integration testleri opt-in ve cache ile sınırlandırılmış.
- Local laptop CPU ile 1 tek yarış simulation hızını benchmark et; performans sınırını ölçmeden “real-time” iddiası ekleme.

---

## 19. ARAŞTIRMA TASARIMI VE BİLİMSEL RAPORLAMA

### 19.1 Ana araştırma soruları

- **RQ1:** Başlangıçtan itibaren sürücü profiliyle koşullanan stratejistler, aynı çevrede profili görmeyen politikalara göre daha karaktere duyarlı ve en azından benzer yarış sonuçları üretebilir mi?
- **RQ2:** Weather/tyre/track verilerinin eklenmesi, farklı yarışlarda expected outcome ve karar kalitesini ne ölçüde artırır?
- **RQ3:** Öğrenilmiş stratejist, güçlü Monte Carlo/MPC baselinelerinden gerçekten daha iyi veya daha hızlı karar veriyor mu?
- **RQ4 (Stage 2):** Opponent modeling ve team cooperation multi-agent strateji kalitesini artırıyor mu?
- **RQ5 (Stage 3):** Sosyal hafıza/ilişki parametreleri, tutarlı uzun vadeli karar farklılıkları üretiyor mu?

### 19.2 Adil deney tasarımı

- Yarış kararının verildiği andaki **bilgi seti** bütün ajanlarda karşılaştırılabilir olmalı.
- Aynı paired weather/incident scenario seti kullan; ancak learned policy'nin training'de gördüğü seed'ler evaluation'da yer almamalı.
- Oracle gerçek geleceği biliyorsa normal baseline gibi gösterme; üst sınır analizi için kullan.
- `historical strategy` gerçek olay referansıdır, mutlaka en iyi/kötü baseline değildir.
- Monte Carlo tahmini ile veri-kaynaklı model tahmini kalibrasyon gücünü ayrı raporla.
- 2024 Monza hem geliştirme hem değerlendirmede kullanılırsa generalization iddiasını sadece bu yarıştan çıkarma.

### 19.3 Rapor şablonu

```text
Experiment ID:
Research question:
Hypothesis:
Dataset manifest and exclusion criteria:
Profile revisions and driver IDs:
Ruleset + engine version:
Baseline policy versions:
Model version + architecture:
Training seeds / Evaluation seeds:
Scenarios (weather, track, car, traffic):
Metrics and confidence intervals:
Profile behavior/strategy visualizations:
Failures and negative results:
Reproducibility commands:
Decision / next experiment:
```

### 19.4 Her yarış stratejisi için “explainability record”

- `decision_tick`, `driver_id`, `driver_profile_revision`, `current_compound`, `weather_obs`, `forecast`, `car`, `tyre_age`, `candidate_actions`, `valid_actions`, `estimated_outcome_distribution`, `chosen_action`, `policy_version`, `driver_intent`, `user_override`.
- Explainability önemli ama eğitimdeki neural modelin kararını LLM'e sonsuz özgürlükle yorumlatma. Önce gerçek `counterfactual estimates`, feature attribution ve constraint flags göster.

---

## 20. RİSK KAYDI VE TEKNİK ALTERNATİFLER

| Risk | Etki | Önlem / fallback |
|---|---|---|
| TUM motoru ara tick/save kabul etmiyor | Çok yüksek | M0'daki adapter spike; gerekirse data-driven event/sector engine; fizik motoru yazma |
| Eski bağımlılıklar Windows/modern Python'da bozuluyor | Yüksek | İzole env, sürüm pinleme, küçük güvenli adapter, fallback |
| Gerçek veriler pit/tyre/weather açısından eksik | Yüksek | NaN audit; estimated parameters/uncertainty; kaynağı gizleme |
| Profile parameters bilimsel kanıt değil | Yüksek | `editorial_assumption`; editör; profil ablasyonları; sürücü hakkında kesin kişilik iddiası yok |
| Learned policy profilleri görmezden geliyor | Yüksek | profile effect tests; condition dynamics ve input ablations; regularization |
| RL simulator exploit | Yüksek | invariants, independent baselines, OOD testing, human inspect |
| Simüle Monza sonucu tarihsel olarak yanlış | Orta | Kalibrasyon raporu; counterfactual kesinlik iddiası yok |
| Pause gecikmesi / save race condition | Çok yüksek | bounded ticks, serialized command queue, atomic snapshots |
| Model değişince save tekrar üretilemiyor | Çok yüksek | immutable manifests, pin/hash, codec migrations |
| UI 2D pist konumları gerçeği yansıtmıyor | Orta | historical vs simulated ayrımı, arc-length interpolation, approximate marking |
| Training laptop'ta ağır | Orta/yüksek | headless vectorized env, curriculum, küçük senaryolar, profiler; cloud sadece sonra |
| Takım logoları/veri lisansı | Yüksek | fan/non-commercial local prototype; source/license doc; ticari yayında hukuk kontrolü |
| Çok geniş kapsam motivasyonu azaltıyor | Yüksek | dikey milestone, 2D demo erken, Stage 1 bitmeden sosyale geçme |
| 2024/2026 F1 kurallarını karıştırma | Yüksek | `RegulationSet` + race-specific roster ve entegrasyon testleri |
| Oyun/ML birlikte debug zorlaşıyor | Orta | headless engine, deterministic replay, model-free baseline, event inspector |

**Sert kural:** Kapsam daraltılabilir; fakat `profile-conditioned Stage 1`, `pause/save/load/fork`, `verified vs assumed data`, `model version pinning` korunur.

---

## 21. CLAUDE CODE OTURUM AKIŞI, GÖREV ŞABLONLARI VE RAPORLAMA

### 21.1 İlk oturum prompt'u (başlatma)

```text
Implement milestone M0 from F1_ALTERNATE_REALITY_MASTER_PLAN.md, not the entire roadmap.

Priority 1: inspect environment; choose supported Python tooling; set up a runnable repo with tests.
Priority 2: implement typed driver profiles with explicit editorial assumptions and a seeded 2024 Monza roster.
Priority 3: fetch/cache and inspect FastF1 Monza 2024 basic data, record missingness and source metadata.
Priority 4: evaluate TUMFTM race-simulation (license, old dependencies, time-step pause/save feasibility). Do not assume it works with 2024; document a go/no-go decision.
Priority 5: build a minimal but truly deterministic headless WorldState + WorldRunner with pause, save, load in a fresh process, resume and fork, plus tests.

Do not produce mock 'done' status, placeholder RL training, social agents or a full complex UI yet. If full data access fails, use clearly labeled synthetic fixtures and still make the headless engine and tests work. At end run tests, list actual files/commands/results, update PROGRESS.md and DECISIONS.md, then stop.
```

### 21.2 Sonraki milestone'a geçiş prompt'u

```text
Read F1_ALTERNATE_REALITY_MASTER_PLAN.md and docs/PROGRESS.md.
Show the current milestone, acceptance tests, unresolved risks and project state.
Implement only the next smallest incomplete milestone task, including code, tests and docs.
Keep world persistence reproducible and driver profiles as actual simulation/training inputs.
Run relevant tests and update docs. Do not implement later-stage features prematurely.
```

### 21.3 Bug / determinism problem prompt'u

```text
Investigate this regression without deleting saves or weakening tests.
Create the smallest reproducible failing test using a fixed seed and profile/model manifest.
Compare serialized snapshots, event cursor, RNG states and model versions before and after save/load.
Fix the root cause; report cause, regression test and limitations.
```

### 21.4 Eğitim talimatı prompt'u

```text
Implement the next Stage 1 profile-conditioned learning experiment from the master plan.
Keep profile vector in observations and verify it affects transitions/behavior.
Compare against rule-based and Monte Carlo baselines with held-out scenarios, paired seeds and no future leakage.
Save model/dataset/profile/engine manifests and report all results, including failures.
Do not mutate an existing user's world or silently replace a pinned model.
```

### 21.5 Claude Code'un her görev sonundaki zorunlu rapor biçimi

```text
Milestone / Task:
Completed:
Files changed:
Commands executed:
Tests: PASS/FAIL counts and exact command:
Evidence/demo output:
Open bugs / known assumptions:
Source/license warnings (if any):
Progress docs updated: yes/no:
Next ONE task:
```

### 21.6 Claude'un asla yapmaması gerekenler

- Veri kaynaklarından API cevabı görmeden sahte 2024 verileri “historical” diye kaydetmek.
- Profil puanlarını gerçek psikometrik ölçüm gibi göstermek.
- “TUM is integrated” deyip sadece GitHub linki README'ye koymak.
- Frontend'de pause ikonu gösterip backend world clock'u ilerletmek.
- Save düğmesini sadece React state/localStorage'a bağlamak.
- Dünya save dosyasının içinde model sürümünü kaydetmemek.
- Monza gerçek yarışının 18. turundan sonra tarihsel pozisyonları sabit tutup buna “alternate reality” demek.
- Sonuç grafiği çizmek için rastgele uydurulmuş ML benchmark sayıları kullanmak.
- Gerçek F1 verisi olmayan parametreleri sessizce 'observed' etiketlemek.
- Stage 3 sohbet ajanlarını Stage 1 yarış optimizasyonunun yerine geçirmek.
- Kullanıcıdan her adımda tekrar aynı kapsam kararlarını istemek.

---

## 22. 10 ADIMLIK İLK GELİŞTİRME CHECKLIST'İ

Aşağıdaki adımlar uygulamaya başlama sırasını açıkça belirler:

- [ ] **1.** Repository/bootstrap. `python -m pytest` en az bir gerçek test geçsin.
- [ ] **2.** `DriverProfile` şeması. Max-style / Hamilton-style ve bütün Monza start kadrosu için kaynak etiketli profile seed.
- [ ] **3.** Headless deterministic two-car world, 1 tick çalıştır.
- [ ] **4.** Pause ve resume. `sim_clock`/state freeze testini geçir.
- [ ] **5.** Snapshot save; backend/pseudoprocess restart sonrası load; eşitlik testi.
- [ ] **6.** Timeline fork, farklı komut, orijinalin bozulmadığının testi.
- [ ] **7.** FastF1 Monza 2024 data fetch/cache/quality manifest.
- [ ] **8.** TUM sim feasibility raporu ve engine adapter/fallback seçimi.
- [ ] **9.** Gerçek dataset ve profile-conditioned pace baseline'ı minimal state transition'a bağla.
- [ ] **10.** Basit Monza haritasını 2D çiz, backend tick ile araç hareketini göster.

**İlk oturumda 10 adımın hepsini yapmak gerekmiyor.** Önemli olan 1–6'nın gerçek testle doğrulanması ve 7–8'in minimum araştırma raporudur.

---

## 23. PROJEYİ BÜYÜTÜRKEN ALINACAK TASARIM KARARLARI

Bu kararları şimdi kesinleştirme; milestone yaklaşınca ölçümle seç.

1. TUM RaceSim adapter mı, hafif event engine mi? → M0 deney sonucu.
2. `simulation_step` 0.25s/1s/sector event mi? → pause latency + CPU throughput.
3. FastF1/OpenF1 hangi event alanları için daha güvenilir? → source completeness raporu.
4. Profil embedding boyutu ve mimarisi? → profile conditioning ablasyonu.
5. RL algoritması MaskablePPO/DQN/başka mı? → eylem uzayı ve benchmark.
6. Ajan tek bir shared policy mi, driver-specific fine tuning mi? → sample efficiency/character fidelity.
7. `WeatherForecast` modeli deterministic scenario prior mı, ayrı ML forecast mi? → veri kalitesi.
8. Kişilik/arzunun öğrenmeye etkisi regularizer mı reward mı? → profile diversity vs race outcome.
9. Opponent model hidden state ve agent communication nasıl olmalı? → Stage 2.
10. Social memory tabanı structured event mi, vector DB mi? → Stage 3'te gerçek ihtiyaç.

---

## 24. KAYNAKLAR, DOKÜMANLAR VE BAĞLANTILAR

Bu kaynaklar **10 Ekim 2026 tarihinde çevrimiçi kontrol edilmiştir**; kurulum sırasında sürüm/lisans ve API'ler yeniden doğrulanmalıdır.

1. TUMFTM Race Simulation (lap-wise, strategy, Monte Carlo, VSE, RL; eski Python 3.8/2014–2019 parametreleri): https://github.com/TUMFTM/race-simulation
2. TUM Race Simulation LGPL-3.0 lisansı: https://github.com/TUMFTM/race-simulation/blob/master/LICENSE
3. TUM Racetrack Database (20+ circuit, Monza centerline & raceline, accuracy caveats): https://github.com/TUMFTM/racetrack-database
4. TUM Racetrack Database LGPL-3.0 lisansı: https://github.com/TUMFTM/racetrack-database/blob/master/LICENSE
5. TUM F1 Timing Database (2014–2019; noncommercial CC BY-NC-SA 3.0): https://github.com/TUMFTM/f1-timing-database
6. FastF1 resmi proje: https://github.com/theOehrly/Fast-F1
7. FastF1 kullanım/dokümantasyon: https://docs.fastf1.dev/
8. OpenF1 docs: https://openf1.org/docs/
9. OpenF1 plans, rate limits and usage terms: https://openf1.org/
10. Jolpica-F1 (Ergast replacement): https://github.com/jolpica/jolpica-f1
11. Jolpica API docs: https://api.jolpi.ca/docs/
12. F1 official — 2024 Italian GP results: https://www.formula1.com/en/results/2024/races/1244/italy/race-result
13. F1 official — Monza 2024 strategy context: https://www.formula1.com/en/latest/article/leclerc-thrills-the-tifosi-to-triumph-at-monza-ahead-of-piastri-and-norris.1aiYZF3rWZp2Q9yQtcuvqV
14. Stable Baselines3 custom envs: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
15. MaskablePPO: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
16. FIA 2026 regulation hub: https://www.fia.com/F126
17. TUM time-discrete race-simulator alternative candidate: https://github.com/heilmeiera/time-discrete-race-simulator
18. TUM lap-time simulation alternative/reference: https://github.com/TUMFTM/laptime-simulation

### Kaynak kullanımı ve ölçülebilir sınırlamalar

- Real historical data → yeniden üretim için uygun snapshot ve kaynak manifesti.
- Assumed simulation dynamics → kalibrasyon ve belirsizlik raporu.
- Driver persona → editoryal simülasyon varsayımı.
- Learned ML weights → model registry ve eğitim protokolü.
- Simulated social events → `fictional alternative reality` etiketi.
- Project open source/commercial dağıtımı → veri, marka, logo, ve kod lisansları ayrı incelenir.

---

## 25. SÖZLÜK / İSİMLENDİRME STANDARTLARI

- **Historical replay:** Gerçek olay kayıtlarını mümkün olduğunca taklit eden, model öğrenmesi olmayan oynatma.
- **Alternate/counterfactual simulation:** Seçilen sapma noktasından sonra simülatörün bağımsız yeniden hesapladığı olası tarih.
- **Canonical state:** Backend'in otoritatif, tek doğru simülasyon durumu.
- **Simulation tick:** Deterministik ve atomik oyun durumu ilerletme adımı.
- **Snapshot:** Bir tick'teki dünya durumunun doğrulanabilir kaydı.
- **Event log:** Gerçekleşen dünya olaylarının sıralı kaydı.
- **Timeline/branch:** Farklı kullanıcı veya AI kararlarıyla ayrılan alternatif olay geçmişi.
- **Driver profile/archetype:** Başlangıçta tanımlanmış, editlenebilir ve AI'a koşul olarak verilen sürücü özellikleri.
- **Driver condition:** Dünya içinde zamanla değişen dinamik state.
- **Learned policy:** RL/supervised/diğer yöntemle eğitim sonucu elde edilmiş eylem seçme modeli.
- **Race strategy:** Pit, compound, pace, stint, hava ve risk kararlarının düzeni.
- **Opponent modeling:** Diğer sürücülerin gizli davranışları üzerine tahmin; Stage 2.
- **Social memory:** Simülasyon içi ilişkileri etkileyen olay/hafıza; Stage 3.
- **No future leakage:** Ajanın henüz yaşanmamış gerçek olayları gözlemlememesi.
- **Profile-conditioned learning:** Politikanın sürücü profilini input olarak görerek ve bu profile bağlı simülasyon/ödül altında öğrenmesi.
- **Oracle:** Geleceğe dair normal ajanın bilmediği bilgiyle çalışan üst sınır stratejisi; adil rakip değil.
- **Seed:** Kontrollü stokastik senaryo üretim anahtarı; tek başına tam determinizm garantisi değildir.
- **Model manifest:** Model/data/profile/engine kodu ve sürüm/parametre metadata'sı.

---

## 26. SON KARAR: NASIL BAŞLAYACAĞIZ?

Bu projenin uzun vadeli hedefi büyük: yaşayan bir F1 dünyası, profil koşullu öğrenen sürücüler, birbirleriyle rekabet eden takımlar ve sosyal stratejiler. **İlk geliştirme birimi ise küçük, doğrulanabilir ve tam kayıtlı bir dünya olacak.**

İlk gün sürücü profillerini tanımlamadan rastgele `if driver == "Max": aggressive` kodu yazmak istemiyoruz. Bunun yerine kimlikten ayrı tipli bir özellik vektörü kullanacağız. Model ilk öğrenme deneyinde o vektörü görecek.

İlk gün monolitik, sadece bir seferlik yarış simülatörü de yazmak istemiyoruz. Oyuncunun her zaman durdurup kaydedebileceği bir `WorldRunner + persistence + timeline` sistemi kuracağız.

İlk gün AI'a gerçek yarışın geleceğini de göstermiyoruz. Canlı dünya şu an gördüğü koşullara göre karar veriyor; alternatif zaman çizgisi başladıktan sonra geleceğini kendisi oluşturuyor.

**İlk somut deliverable:** Monza 2024 verilerini değerlendiren, en az iki farklı başlangıç sürücü profilinin karar/sürüş durumunu taşıyan, 2–20 araçla ilerleyebilen headless yarış prototipi. Bu prototipin **pause → save → process exit → restore → resume → fork** testleri geçecek. Ardından 2D arayüz ve öğrenen stratejist eklenecek.

> Claude Code: Bu belgeyi okuyup geliştirmeye M0'dan başla. Önce çalışan kod ve test, sonra bir sonraki milestone. **Uzun vadeli vizyonu koru; ama hiçbir aşamayı gerçekte yapılmamışken tamamlandı olarak işaretleme.**
