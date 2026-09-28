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
выберутся те же64+192координаты.

## 4. Глобальная голова

```bash
python train_global_metric.py --self-test
python train_global_metric.py \
  --training-features replay/training_features \
  --training-rows training/training_cpu_rows.jsonl \
  --center-weights replay/chel_v1/model.npz \
  --output replay/global_metric
```

По умолчанию проверяются2500×768признаков и400ID — контракт выбранного опыта.
Для другого явно заданного набора CLI допускает `--expected-rows`,
`--expected-ids`, `--dimension`; это уже другой эксперимент.
Ridge0.1 зафиксирован. Голова не читает test или validation.

## 5. Переход к runtime

Квалифицированный runtime использует `chel_v1.npz`, `chel_v2.npz`,
`compression.npz`, `global_metric.npz` и pretrained backbone/detector.
Не копируйте новую голову в старую поставку автоматически. Сначала проверьте:
размерность1024, сохранение выбранных64/192блоков, нормировку, векторную parity,
качество на frozen validation и калибровку порога для новой модели.

Создайте новый `manifest.json` с фактическими именами, backbone, threshold,
`global_metric` и SHA256 всех весов. Файл становится fingerprint новой модели;
ей нужна отдельная галерея. Старый порог0.3633712828159332 воспроизводит выбранную
модель и не переносится на любое новое обучение без проверки.

Доказанный replay текущего решения — только global-head по сохранённым train
features и свежий инференс восьми изображений. Полный повтор всех стадий
обучения заново перед публикацией не выполнялся.
