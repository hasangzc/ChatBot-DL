import pickle
import random
import os
import sys
from model import model
from datapreprocessing import data, words
from prod import bow
import numpy as np
import discord


def restart_bot():
    os.execv(sys.executable, ["python"] + sys.argv)


class MyClient(discord.Client):
    async def on_ready(self):
        print("Logged in as")
        print(self.user.name)
        print(self.user.id)
        print("------")

    async def on_message(self, message):
        json_file = data
        print("HHG-BOT ile konuşmaya başla. quit yazarsan gider:)")
        inp = message.content
        p = bow(inp, words)
        results = model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(results)]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            labels = pickle.load(open("labels.pkl", "rb"))
            return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
        list_of_intents = json_file["intents"]
        for i in list_of_intents:
            ints = return_list
            tag = ints[0]["intent"]
            result = "null"
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                await message.channel.send(result.format(message))
                restart_bot()


client = MyClient()
client.run("OTYwODc2NTc0OTkzOTYxMDEw.Ykw0Cw.GDZEjjEKkeE0Dwm8Lxt7gwgjfCg")
