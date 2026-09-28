# Запуск и эксплуатация

## Состав

`docker-compose.yml` запускает frontend, backend и database. Образ один,
процессы разные; модель загружается только backend. Исходники и веса
подключаются из текущего checkout. Нельзя запускать старый runtime-образ
без актуальных bind mounts и считать это новой моделью.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `FALCON_IMAGE` | `falcon-chel:final-20260928` | Локально подготовленный образ; для Release задайте `falcon-chel:public-1bf26a1` |
| `FALCON_PORT` | `8799` | Порт frontend, bind только 127.0.0.1 |
| `FALCON_DATA_DIR` | `./data` | Каталог с разрешённым вам dataset.zip |
| `FALCON_OUTPUT` | `/results` | Связанные с этой моделью результаты внутри контейнера |

Модель и DB работают во внутренней Docker-сети. Frontend также имеет edge-сеть;
для него полная сетевая изоляция не заявляется. Ни API 8000, ни DB 8002 не
публикуются на хосте. Галерея — SQLite в именованном volume.

## Первый запуск

1. Получите файлы по [контракту](REPRODUCIBILITY.md). Сам clone весов не содержит.
2. Для готовой среды Linux x86-64 скачайте `falcon-chel-public-cpu-v1.tar.gz`
   из Release и проверьте его SHA256. Задайте
   `export FALCON_IMAGE=falcon-chel:public-1bf26a1`.
3. Запустите `sh start_offline.sh /path/to/falcon-chel-public-cpu-v1.tar.gz`:
   скрипт загрузит образ при его отсутствии; Compose без pull/build ожидает healthy.
   Сохраняйте `FALCON_IMAGE` для всех следующих команд. Альтернатива — сборка
   `docker build -t falcon-chel:local .` и `export FALCON_IMAGE=falcon-chel:local`.
4. Проверьте `/health` и `/docs`. Проверьте реальный поиск на своих данных.

```bash
docker compose ps
docker compose logs --tail=50 database backend frontend
docker compose restart
docker compose down
```

Обычный `down` сохраняет volume; `down -v` удаляет его — для обычного выключения
этот флаг не нужен. При смене модели используйте отдельную галерею/volume.
Несовместимый fingerprint останавливает импорт, а не молча заменяет происхождение.

## Запуск рядом с другой установкой

Compose по умолчанию использует project `falcon-v5`. Два checkout с этим именем
могут управлять одними контейнерами и volume. Для отдельной установки задайте
**и новый project, и свободный порт**; сохраняйте их для всех следующих команд:

```bash
export COMPOSE_PROJECT_NAME=falcon-local-audit
export FALCON_PORT=18801
export FALCON_DATA_DIR=/absolute/path/to/authorized/data
sh start_offline.sh
docker compose ps
```

Переменная `COMPOSE_PROJECT_NAME` имеет приоритет над `name:` в YAML.
При таком имени создаётся отдельный volume; прежняя галерея не импортируется
из другого volume автоматически. Не используйте `docker compose down` с именем
чужой работающей установки. Проверка `tests/qualify_runtime.py` вообще не требует
Compose и использует собственную временную SQLite.

## Пустая и подготовленная галерея

Без `outputs/rows.json` backend стартует с DB и допускает загрузку своих
объектов через `/gallery`. Конкурсные готовые примеры тогда недоступны.
Если инференс сохранён в `outputs/new_run`, перед запуском задайте
`export FALCON_OUTPUT=/results/new_run`. Подготовленный импорт требует не только embeddings: проверяются
`model_manifest.json`, `rows.json`, `pooling_weights.npy`, `redactions.json`
и их связи с `report.json`. Нельзя копировать новый manifest к старым векторам.

## Внешнее демо

Compose намеренно слушает localhost. Для доступа извне используйте отдельный
ограниченный reverse proxy к frontend. Оставьте upstream фиксированным,
закройте запись в галерею, ограничьте размер запроса, параллелизм и время.
Не публикуйте архив данных, DB, каталог весов или файловую систему сервера.
Включайте HTTPS там, где его обеспечивает ваш домен/прокси.

Одно открытие порта на роутере не доказывает внешний доступ: проверьте URL
из другой сети. Адреса конкретной установки и credentials не входят в Git.
Репозиторий не обещает круглосуточный публичный endpoint или промышленный SLA.

## Границы измерений

Проверенная backend-конфигурация: два CPU, 6 GiB; DB/frontend по 0.5 CPU / 256 MiB.
Это настроенные лимиты, не минимальные требования. Проверенная галерея содержит 750 изображений.
Задержка объяснения отличается от encode/search; отдельный нагрузочный
профиль для многих одновременных операторов не измерялся.
