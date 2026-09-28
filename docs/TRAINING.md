# Обучение из исходников

Команды описывают исследовательский протокол CHEL. Они требуют полученных
законным образом изображений, собственного train/calibration/validation split
и масок; эти приватные входы не включены в Git. Смена данных или split меняет
результат, поэтому это не обещание получить прежние цифры на произвольном наборе.

## 1. Зафиксировать входы

Подготовьте `training/training_cpu_rows.jsonl`, `heldout_cpu_rows.jsonl`,
`training_masks.json`, `heldout_masks.json` и `training/splits_v2/` по форматам
CLI и исходников. Разделяйте ID и связанные общие кадры. Храните hashes входов.
DINO backbone размещается в `weights/backbone.safetensors`.

Все JSONL-файлы содержат по одному объекту JSON в строке, без внешнего массива.
Пример ниже синтетический: он показывает поля, а не реальный конкурсный ID.

```json
{"image_id":"sample-car-001","source_row":0,"vehicle_id":"vehicle-a","camera_id":"camera-1","x":10,"y":20,"w":100,"h":80}
```

`source_row` — индекс строки исходного CSV с нуля, без заголовка. `image_id`
должен совпадать с CSV и изображением в архиве. Для train обязательны
`vehicle_id` и `camera_id`; для encoder достаточен `image_id`, но сохранённые
feature rows автоматически получают исходный `source_row`. Глобальная голова
проверяет точное соответствие этого индекса подготовленным metadata.

В фактически полученном train CSV присутствовал `camera_id`, хотя описание
ТЗ говорило об отсутствии камерных metadata. В test CSV этого поля нет.
Здесь train camera_id используется для формирования пар и локального
протокола оценки; при inference камера, время и география не нужны и
не передаются модели. Если доступный вам train не имеет camera_id, не
придумывайте значения: описанный межкамерный протокол так не воспроизводится.


`training/splits_v2/` содержит четыре JSONL-файла: `calibration_query.jsonl`,
`calibration_gallery.jsonl`, `validation_query.jsonl`, `validation_gallery.jsonl`.
В них нужны `image_id`, `vehicle_id`, `camera_id`; BBox/source_row сохраняйте для
аудита. `heldout_cpu_rows.jsonl` должен охватывать объединение их изображений.
В query нужны как ID, присутствующие в gallery, так и отдельные отсутствующие ID
для калибровки отказа. Совпадения одной камеры исключаются evaluator.
Нельзя переносить обучающие ID или связанные общие кадры в held-out выборки.

Формат `training_masks.json` и `heldout_masks.json` отличается от reviewed-ledger
инференса: это отображение image ID в прямоугольники **исходного полного кадра**:

```json
{"coordinate_system":"full_frame_xyxy","reviewed":false,"images":{"sample-car-001":[[25,65,60,80]]}}
```

Одна запись нужна для каждого выбранного изображения, даже если список пуст.
Не ставьте `reviewed:true` без реальной проверки. Формальная валидность файла
не гарантирует, что все номерные области закрыты. Реестр для `inference.py --reviewed-redactions` имеет другой контракт в `falcon/reviewed_redactions.py`.

Для воспроизведения старого опыта используются автоматические masks и флаг
`--allow-unreviewed-masks`; это **не** подтверждение отсутствия признаков номера.
Для новых приемочных результатов маски должны быть действительно проверены.

## 2. Извлечь признаки на CPU

```bash
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -m falcon.encode --archive data/dataset.zip --csv train.csv \
  --weights weights/backbone.safetensors --timm-name vit_base_patch16_dinov3.lvd1689m \
  --redactions training/training_masks.json --selected-rows training/training_cpu_rows.jsonl \
  --output replay/training_features --device cpu --threads 2 --batch-size 8 --dense \
  --allow-unreviewed-masks
python -m falcon.encode --archive data/dataset.zip --csv train.csv \
  --weights weights/backbone.safetensors --timm-name vit_base_patch16_dinov3.lvd1689m \
  --redactions training/heldout_masks.json --selected-rows training/heldout_cpu_rows.jsonl \
  --output replay/held_features --device cpu --threads 2 --batch-size 8 --dense \
  --allow-unreviewed-masks
```

RGB, маскирование, crop по BBox, стандартный timm transform256×256, avg pooling.
Локальные признаки в сохранённом протоколе проходят промежуточный float16.

## 3. Обучить локальные ветви и сжатие

```bash
python -m falcon.evidence_ledger fit \
  --training-features replay/training_features --training-rows training/training_cpu_rows.jsonl \
  --held-features replay/held_features --splits training/splits_v2 \
  --output replay/chel_v1 --seed 20260924 --utility-dim 64 --support-dim 192 \
  --hidden-dim 128 --positive-k 1 --negative-k 1
python -m falcon.evidence_ledger fit \
  --training-features replay/training_features --training-rows training/training_cpu_rows.jsonl \
  --held-features replay/held_features --splits training/splits_v2 \
  --output replay/chel_v2 --seed 20260924 --utility-dim 64 --support-dim 768 \
  --hidden-dim 128 --positive-k 2 --negative-k 3 --orthogonal-support
python -m falcon.compress_ledger fit \
  --training-features replay/training_features --held-features replay/held_features \
  --v1-model replay/chel_v1/model.npz --v2-model replay/chel_v2/model.npz \
  --splits training/splits_v2 --output replay/compression --seed 20260925 \
  --global-threshold 0.20663335919380188
```

PCA учится на train; выбор размера и параметров использует calibration.
Сохранённые скрипты поддерживают gate/evaluation, смотрите `--help`.
`selection.json` содержит выбранную конфигурацию и решение keep/reject.
Нельзя автоматически продвигать reject или предполагать, что на других данных
выберутся те же 64 + 192 координаты.

## 4. Глобальная голова

```bash
python train_global_metric.py --self-test
python train_global_metric.py \
  --training-features replay/training_features \
  --training-rows training/training_cpu_rows.jsonl \
  --center-weights replay/chel_v1/model.npz \
  --output replay/global_metric
```

По умолчанию проверяются 2500 × 768 признаков и 400 ID — контракт выбранного опыта.
Для другого явно заданного набора CLI допускает `--expected-rows`,
`--expected-ids`, `--dimension`; это уже другой эксперимент.
Ridge 0.1 зафиксирован. Голова не читает test или validation.

## 5. Переход к runtime

Соответствие результатов команд именам runtime:

| Результат обучения | Имя в новом каталоге весов |
|---|---|
| `replay/chel_v1/model.npz` | `chel_v1.npz` |
| `replay/chel_v2/model.npz` | `chel_v2.npz` |
| `replay/compression/model.npz` | `compression.npz` |
| `replay/global_metric/global_metric.npz` | `global_metric.npz` |

Перед копированием проверьте `replay/compression/selection.json`: нужен
`decision:"keep"`, `winner.family:"separate_pca"`, `v1_dimension:64` и
`v2_dimension:192`. Обучающий CLI исследует и другие конфигурации; текущий
runtime предназначен именно для этих двух раздельных блоков. `keep` для другой
семьи или размерности не делает её совместимой с этим runtime.
Локальные fit-команды сохраняют `frozen_selection.json` и `fit_report.json`;
команда сжатия — `selection.json`. Это разные файлы.

Квалифицированный runtime использует `chel_v1.npz`, `chel_v2.npz`,
`compression.npz`, `global_metric.npz` и pretrained backbone/detector.
Не копируйте новую голову в старую поставку автоматически. Сначала проверьте:
размерность 1024, сохранение выбранных 64/192 блоков, нормировку, векторную parity,
качество на frozen validation и калибровку порога для новой модели.

Создайте новый `manifest.json` с фактическими именами, backbone, threshold,
`global_metric` и SHA256 всех весов. Файл становится fingerprint новой модели;
ей нужна отдельная галерея. Старый порог 0.3633712828159332 воспроизводит выбранную
модель и не переносится на любое новое обучение без проверки.

Доказанный replay текущего решения — только global-head по сохранённым train
features и свежий инференс восьми изображений. Полный повтор всех стадий
обучения заново перед публикацией не выполнялся.

Формат runtime manifest и полная проверка предоставленных артефактов описаны в
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Полного автоматического экспорта
произвольного заново обученного решения в готовую поставку здесь нет: принятие
конфигурации, калибровка итогового KISSME-вектора и отдельный manifest обязательны.
CLI `--help` и синтетические контроли проверяют доступность команд, а не полный
повтор обучения или качество новой модели.
