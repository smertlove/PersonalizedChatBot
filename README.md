# PersonalizedChatBot

### Задание

1) **Основная задача:** нужно придумать бейзлайны для каждого из 3 агентов и собрать простейший прототип системы. Можно использовать любые технологии и решения. Ниже даны рекомендации по реализации.
     - Обсуждалось, что агент извлечения фактов из контекста может возвращать триплеты subj + pred + obj. Тогда его можно построить как seq2seq модель, либо как правила Spacy.
     - Агент извлечения фактов из базы можно построить как комбинацию поискового индекса и некоторой системы ранжирования. Но нужно сделать поправку на то, что документы в базе будут очень маленькие (триплеты).
     - В качестве агента генерации ответа может выступать некоторая LLM.
2) **Дополнительная задача:** нужно придумать, как оценить полученные агенты.

###  Запуск

- Убедитесь, что docker compose установлен: `docker compose version`
- Билд: `sudo docker compose build`
- Запуск: `sudo docker compose run`

### Путеводитель по проекту:

Проект разворачивается в виде микросервисов, что позволяет использовать более свободно выбирать зависимости для каждого компонента. 

- Клиент находится тут: http://0.0.0.0:5001/
- phpMyAdmin находится тут: http://0.0.0.0:6660/
- База данных находится тут: http://0.0.0.0:3306/
- Проверить, что остальные сервисы поднялись:
     - http://0.0.0.0:5002/ -- query_retriever
     - http://0.0.0.0:5003/ -- db_retriever
     - http://0.0.0.0:5004/ -- reply_generator
 
### N.B.

- База данных создается на старте контейнера, но только один раз (она сохранится в памяти, и при повторном пуске код не отработает). Поэтому не забывайте делать `sudo docker container prune`, если будете вносить правки в базу.
