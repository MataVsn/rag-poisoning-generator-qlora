# Фактическое развёртывание — 16 ноября 2026

Проект: `/workspace/rag-genai-endterm` на арендованном инстансе. GPU: RTX 5090, 31.36 GiB доступной видеопамяти. Образ содержит PyTorch 2.14.1+cu132. Для экономии диска создано окружение `.venv` с `--system-site-packages`, которое использует этот PyTorch; остальные обучающие библиотеки установлены в окружение проекта.

Это отличается от стартового bootstrap с PyTorch 2.7.1/cu128. Фактические версии сохраняются в `outputs/remote/installed-packages.json`, версия CUDA и override bitsandbytes — в итоговом training manifest.

В `configs/runtime.env` на сервере заданы:

```sh
export BNB_CUDA_VERSION=130
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1
```

bitsandbytes 0.48.1 не содержит бинарник CUDA 13.2; используется его сборка CUDA 13.0. Реальная проверка NF4 forward и CUDA matmul прошла. Начальные шаги QLoRA также прошли: обучаются 11 796 480 параметров (0.2924% модели), наблюдалось около 6360 MiB занятой GPU-памяти. Примерно 2.5–3 секунды на optimizer step — предварительное наблюдение, итоговая скорость и пиковая память будут в training manifest. Полный training pipeline проверяется отдельным smoke-прогоном, одного preflight для этого недостаточно.

Подключение с ноутбука:

```sh
ssh -i ~/.ssh/<key> -p <PORT> root@<GPU_HOST>
cd /workspace/rag-genai-endterm
sh scripts/remote_train.sh status
sh scripts/remote_train.sh logs
```

Для ручных команд Python сначала выполнить `. ./configs/runtime.env`, затем использовать `.venv/bin/python`. Скрипт remote_train.sh загружает настройки сам.

На инстансе нет смонтированного сохраняющегося host volume. Stop/start сохраняет контейнер, destroy/recycle уничтожает данные. Адаптер и результаты нужно скачать на ноутбук до удаления инстанса. Доступный диск контейнера — 16 GiB; новый экземпляр PyTorch и CUDA поверх существующих намеренно не устанавливаются.

Не выполнять `bootstrap_gpu.sh` повторно на этом инстансе: он предназначен для установки отдельного PyTorch на новом сервере. Здесь окружение уже подготовлено с учётом ограниченного диска.

## Проверенный smoke-прогон

20 optimizer steps завершились с exit_code=0: train runtime 54.1976 s, peak allocated 4.8779 GiB, peak reserved 5.4766 GiB. Адаптер успешно загрузился для генерации одного отложенного validation-примера; непустой документ получен. Это проверка работоспособности, не оценка ASR и не итоговая модель. После неё запущено полное обучение на mixed train, две эпохи.
