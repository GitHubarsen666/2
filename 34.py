#Об’єктно-орієнтоване програмування (ООП)
class User:
    def __init__(self, name):
        self.name = name
        self.activity_log = []


    def record_activity(self, action):
        self.activity_log.append(action)


class Book:
    def __init__(self, title):
        self.title = title


# Приклад використання
user = User("Анна")
book = Book("Штучний інтелект у бібліотеках")
user.record_activity(f"Переглянула книгу: {book.title}")
print(user.activity_log)

#Агентно-орієнтоване програмування (АОП)
import random


class LibraryAgent:
    def __init__(self, name):
        self.name = name


    def analyze_user_activity(self, activity_score):
        if activity_score < 30:
            return f"{self.name}: користувач має низьку активність. Надіслати рекомендації."
        else:
            return f"{self.name}: активність користувача стабільна."


agent = LibraryAgent("Агент_Моніторингу")
print(agent.analyze_user_activity(random.randint(10, 60)))
