from datetime import datetime, date

def get_current_time():
    now = datetime.now()
    today = date.today()
    current_time = str(now.strftime("%H:%M:%S"))
    current_time = current_time[0:2] + 'h' + current_time[3:5] + 'm'
    return str(today) + '-' + current_time

