from twilio.rest import Client


account_sid = 'XXX'
auth_token = 'XXX'

def send_sms(score, happiness, calm, sadness, anger, fear, name):
    client = Client(account_sid, auth_token)
    dict = {
        "happiness": happiness,
        "calm": calm,
        "sadness": sadness,
        "anger": anger,
        "fear": fear,
    }
    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}
    keys = list(dict.keys())
    if score <= 50:
        message_body = "Hello, your friend " + name + " is not doing alright. We wanted to update you with a summary" \
                                                      "of his/her recently experienced prominent emotions which include " + \
                       keys[-1] + " and " + keys[-2] + ". You should reach out as soon as you can."

    else:
        message_body = "Hello, your friend " + name + " is doing well. We wanted to update you with a summary" \
                                                      "of his/her recently experienced prominent emotions which include " + \
                       keys[-1] + " and " + keys[-2] + ". Keep supporting your friend's mental well-being!"

    message = client.messages \
                .create(
                     body=message_body,
                     from_='+14378002075',
                     to='+16476857747'
                 )

    print(message.sid)


#send_sms(60, 30, 40, 50, 60, 70, "Don")