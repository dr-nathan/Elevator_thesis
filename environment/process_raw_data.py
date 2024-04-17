import copy
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def import_data(file_path):

    # Work Fine:
    # woensdag_donderdag, vrijdag_weekend, 2_weken_november
    # 27_10tot31_10_2023, 27_10tot03_11_2023, 27_10tot06_11_2023(copies of 2 weken november)
    # nathan_onderbroken (copy of 2 weken november)

    # Dont work:
    # 31_07_2023_08_08_2023, 13juli_17juli, Nathan.txt, laatste_voor_crash, tot_en_met_27_juli

    # largest files:
    # 27_10tot06_11_2023.txt, 2weken_november.txt

    # train:

    print(f'#### importing {file_path} ####')

    colnames = ["time", "node", "lift", "event", "floor", "door", "on_off", 'extra']
    data = pd.read_csv(file_path, header=None, encoding='unicode_escape', names=colnames)
    # remove rows with stuff in extra column
    data = data[data['extra'].isna()]
    # then drop column
    data = data.drop(columns=['extra'])

    # strip last part of time: everything after (
    data["time"] = data["time"].str.split("(").str[0]
    data["time"] = pd.to_datetime(data["time"], format="%d.%m.%Y-%H:%M:%S.%f", errors='coerce')

    # measure amount of NaT values
    print('percentage of NaT:', (data['time'].isna().sum() / len(data['time']) * 100), '%')
    # drop NaT rows
    data = data[data['time'].notna()]

    print('len before filter: ', len(data))

    # filter out Bus errors
    data = data[data["node"] != " Bus error: Bit Stuff Error"]
    # check percentage of data that is NaN
    print('percentage of NaN:', (data['event'].isna().sum() / len(data['event']) * 100), '%')  # 0.003 %
    # remove those rows
    data = data[data['event'].notna()]
    # get rid of Node and door columns
    data = data.drop(columns=["node", "door"])

    # filter out request hall calls
    data = data[data["event"] != " Request all active hall calls"]
    # filter out 'no floor' events
    data = data[~data["event"].str.contains("no floor")]
    # filter out hall call acknoledgements
    data = data[~data["event"].str.contains("Ack. Standard hall call")]
    # filter out Position events
    data = data[~data["event"].str.contains("Position =")]
    # filter out "all floors" events
    data = data[~data["floor"].str.contains("all floors", na=False)]

    # add column with position
    position = [2] * 6
    for i, row in data.iterrows():
        # update positions
        if 'Position indicator floor' in row.event:
            position[int(row.lift.split(' ')[-1]) - 1] = int(row.event.split(' ')[-1])
        # add position to row
        if (('-' not in row.lift) &
                (("car call" in row["event"]) |
                 (("hall call" in row["event"]) & (row["on_off"] == " on")) |
                 (("Arrival indication" in row["event"]) & (row["on_off"] == " on")))):
            try:
                data.at[i, 'position'] = position[int(row.lift.split(' ')[-1]) - 1]
            except ValueError:
                print('error at row: ', i)
                print(row)
                data.at[i, 'position'] = input('position: ')
            except IndexError:
                print('error at row: ', i)
                print(row)
                data.at[i, 'position'] = input('position: ')
        else:
            data.at[i, 'position'] = None

    # only keep hall & car calls
    data = data[((data["event"].str.contains("car call")) |
                 ((data["event"].str.contains("hall call")) & (data["on_off"] == " on")) |
                 ((data["event"].str.contains("Arrival indication")) & (data["on_off"] == " on")))]
    # ((data['event'].str.contains('Position indicator')) & (data['on_off'] == ' on')))]

    # save to txt
    data.to_csv(f'data/filtered/{file}', index=False)
    print(f'saved to data/filtered/{file}')

    print('len after filter: ', len(data), '\n')


class PassengerData:

    def __init__(self, arrival_time=None, floor=None, direction=None):
        self.arrival_time = arrival_time
        self.floor = floor
        self.direction = direction
        self.destination = None
        self.time_waited = None

    def __repr__(self):
        return f'passenger(arrival_time={self.arrival_time}, ' \
               f'floor={self.floor}, ' \
               f'direction={self.direction}, ' \
               f'destination={self.destination}, ' \
               f'time_waited={self.time_waited})'


def register_passengers(file_path):
    data = pd.read_csv(file_path, header=0)
    data['floor'] = data['floor'].str.split(' ').str[-1].astype(int)
    # correct for floors starting at 1
    data['floor'] = data['floor'] - 1
    data['position'] = data['position'] - 1
    data["time"] = pd.to_datetime(data["time"])

    elev_occupants = defaultdict(list)  # list of passenger objects currently in elev
    elev_occupants_history = defaultdict(list)  # history of elev occupants

    floor_buttons = np.zeros((2, 17))  # up/down x floors NOTE: not used for now
    elev_buttons = np.zeros((17, 6))  # floor x elevator
    elev_serving_floor = np.full(shape=(6,), fill_value=2)  # last floor that the elev stopped at

    hall_calls = defaultdict(list)
    matched_passengers = []
    extra_hall_calls = []
    ghosts = []
    was_already_off = []
    primary_passengers = 0
    copied_passengers = 0
    double_presses = 0
    double_presses_hall = 0
    double_presses_hall_still_on = 0
    wrong_buttons = 0
    car_calls_at_current_floor = 0
    arrival_pos_not_floor = 0
    no_passenger_in_elev = 0
    total_car_calls = 0
    total_car_calls_off = 0
    total_hall_calls = 0

    for ix, row in data.iterrows():

        # update last stop of elev
        if 'Arrival indication' in row.event:
            # update floor of last stop
            elev_serving_floor[int(row.lift.split(' ')[-1]) - 1] = row.floor
            # set relevant floor button to 0
            if 'upward' in row.event:
                floor_buttons[0, row.floor] = 0
            elif 'downward' in row.event:
                floor_buttons[1, row.floor] = 0

        if 'hall call' in row.event:
            total_hall_calls += 1
            if 'upward' in row.event:
                # if theres already a hall call to go up, this is just a repeat
                if hall_calls[str(row.floor) + str('up')]:
                    double_presses_hall += 1
                if floor_buttons[0, row.floor] == 1:
                    double_presses_hall_still_on += 1
                floor_buttons[0, row.floor] = 1
                # register passenger
                passenger = PassengerData(arrival_time=row.time, floor=row.floor, direction='up')
                hall_calls[str(row.floor) + str('up')].append(passenger)
            elif 'downward' in row.event:
                if hall_calls[str(row.floor) + str('down')]:
                    double_presses_hall += 1
                if floor_buttons[1, row.floor] == 1:
                    double_presses_hall_still_on += 1
                floor_buttons[1, row.floor] = 1
                passenger = PassengerData(arrival_time=row.time, floor=row.floor, direction='down')
                hall_calls[str(row.floor) + str('down')].append(passenger)

        if ('car call' in row.event) & ('on' in row.on_off):

            total_car_calls += 1

            # if the button was already pressed, this is just a repeat
            if elev_buttons[row.floor, int(row.lift.split(' ')[-1]) - 1] == 1:
                double_presses += 1
                continue

            # this is where the passenger got in
            last_stop = elev_serving_floor[int(row.lift.split(' ')[-1]) - 1]

            if row.floor > last_stop:  # for upward calls

                # purge all passengers that have been waiting more than 5 minutes
                hall_calls[str(last_stop) + str('up')] = [passenger for passenger in
                                                          hall_calls[str(last_stop) + str('up')] if
                                                          (row.time - passenger.arrival_time).total_seconds() < 300]

                hall_calls_len = len(hall_calls[str(last_stop) + str('up')])

                # if there are registered hall calls at last stop, we assume those are matched to the car call
                if hall_calls_len > 0:
                    # get last passenger from hall calls, remove from hall calls
                    passenger = hall_calls[str(last_stop) + str('up')].pop(0)
                    extra_pass = hall_calls.pop(str(last_stop) + str('up')) if hall_calls_len > 0 else []
                    primary_passengers += 1

                # if no pending hall calls at last elev stop, has already been matched -> append to matched passenger
                else:
                    if len(elev_occupants[row.lift]) > 0:
                        passenger = copy.deepcopy(elev_occupants[row.lift][-1])
                        extra_pass = []
                        copied_passengers += 1
                    else:
                        # check if people perhaps pressed the wrong direction button.
                        if len(hall_calls[str(last_stop) + str('down')]) > 0:
                            passenger = hall_calls[str(last_stop) + str('down')].pop(0)  # remove only 1
                            wrong_buttons += 1
                            extra_pass = []
                            copied_passengers += 1
                        else:
                            # in this case, no hall call registered but also no passenger in elev. Ghost!
                            passenger = PassengerData(arrival_time=row.time, floor=row.floor, direction='up')
                            ghosts.append(passenger)
                            continue

            elif row.floor < last_stop:  # for downward calls

                hall_calls[str(last_stop) + str('down')] = [passenger for passenger in
                                                            hall_calls[str(last_stop) + str('down')] if
                                                            (row.time - passenger.arrival_time).total_seconds() < 300]

                hall_calls_len = len(hall_calls[str(last_stop) + str('down')])

                if hall_calls_len > 0:
                    passenger = hall_calls[str(last_stop) + str('down')].pop(0)
                    extra_pass = hall_calls.pop(str(last_stop) + str('down')) if hall_calls_len > 0 else []
                    primary_passengers += 1

                else:
                    if len(elev_occupants[row.lift]) > 0:
                        passenger = copy.deepcopy(elev_occupants[row.lift][-1])
                        extra_pass = []
                        copied_passengers += 1
                    else:
                        if len(hall_calls[str(last_stop) + str('up')]) > 0:
                            passenger = hall_calls[str(last_stop) + str('up')].pop(0)  # remove only 1
                            wrong_buttons += 1
                            extra_pass = []
                            copied_passengers += 1
                        else:
                            passenger = PassengerData(arrival_time=row.time, floor=row.floor, direction='down')
                            ghosts.append(passenger)
                            continue
                        # raise ValueError('no original hall calls found for this car call')

            else:
                car_calls_at_current_floor += 1
                continue
                # raise ValueError('car call cant be at same pos as elevator')

            elev_buttons[row.floor, int(row.lift.split(' ')[-1]) - 1] = 1
            elev_occupants[row.lift].append(passenger)
            # assign dest to passenger
            passenger.destination = row.floor
            passenger.time_waited = (row.time - passenger.arrival_time).total_seconds()
            matched_passengers.append(passenger)
            for pass_ in extra_pass:
                pass_.destination = None
                pass_.time_waited = (row.time - pass_.arrival_time).total_seconds()
            extra_hall_calls.extend(extra_pass)

            elev_occupants_history[row.lift].append(len(elev_occupants[row.lift]))

        if ('car call' in row.event) & ('off' in row.on_off):

            total_car_calls_off += 1

            if row.position != row.floor:
                arrival_pos_not_floor += 1
                elev_buttons[row.floor, int(row.lift.split(' ')[-1]) - 1] = 0
                continue
                # raise ValueError('car call off at wrong floor')

            if len([passenger for passenger in elev_occupants[row.lift] if passenger.destination == row.floor]) == 0:
                no_passenger_in_elev += 1
                continue

            if elev_buttons[row.floor, int(row.lift.split(' ')[-1]) - 1] == 0:
                was_already_off.append(row)

            elev_buttons[row.floor, int(row.lift.split(' ')[-1]) - 1] = 0
            [elev_occupants[row.lift].remove(passenger) for passenger
             in elev_occupants[row.lift] if passenger.destination == row.floor]

    print(f'\n\n #### file = {file_path} #### \n\n ')

    print('passengers with destination: ', len(matched_passengers))
    print('passengers from hall calls: ', primary_passengers)
    print('passengers from copied passengers: ', copied_passengers, '\n')

    print('total hall calls: ', total_hall_calls)
    print('hall calls while already people waiting: ', len(extra_hall_calls))
    print('double hall calls or shortly after departure: ', double_presses_hall)
    print('hall calls while button still on: ', double_presses_hall_still_on, '\n')

    print('total car calls:', total_car_calls)
    print('total car calls off:', total_car_calls_off)
    print('car calls at current floor: ', car_calls_at_current_floor)
    print('double car call presses: ', double_presses)
    print('car call off signal but was already off: ', len(was_already_off), '\n')

    print('ghosts: car calls with no hall calls or people in elev -> no arrival time', len(ghosts))
    print('wrong direction buttons: ', wrong_buttons, '\n')

    print('no passenger in elev at arrival: ', no_passenger_in_elev)
    print('arrival pos not floor: ', arrival_pos_not_floor, '\n')

    return matched_passengers, extra_hall_calls, ghosts, double_presses, elev_occupants_history


# from raw to filtered
for file in os.listdir("data/raw"):
    # skip DS_Store stuff and other hidden files
    if file[0] == '.':
        continue
    file_path = os.path.join("data/raw", file)
    import_data(file_path)

# from filtered to JSON
for file in os.listdir("data/filtered"):
    # skip DS_Store stuff and other hidden files
    if file[0] == '.':
        continue
    file_path = os.path.join("data/filtered", file)
    matched_passengers, unmatched_passengers, ghosts, double_presses, elev_occupants_history = (
        register_passengers(file_path))

    # merge matched and unmatched
    merged = sorted((matched_passengers + unmatched_passengers), key=lambda x: x.arrival_time)
    # save as JSON
    with open(f'data/JSON/{file[:-4]}.json', 'w') as f:
        # convert Timestamps to strings
        local = copy.deepcopy(merged)
        for passenger in local:
            passenger.arrival_time = passenger.arrival_time.strftime('%Y-%m-%d %H:%M:%S')
        json.dump([passenger.__dict__ for passenger in local], f, indent=4)

    # distribution of time waited
    time_waited = [passenger.time_waited for passenger in matched_passengers]
    time_waited_um = [passenger.time_waited for passenger in unmatched_passengers]
    time_waited = [300 if t > 300 else t for t in time_waited]
    time_waited_um = [300 if t > 300 else t for t in time_waited_um]
    fig, ax = plt.subplots()
    ax.hist(time_waited, bins=150, alpha=0.5, label='matched')
    ax.hist(time_waited_um, bins=150, alpha=0.5, label='unmatched')
    ax.legend(loc='upper right')
    ax.set_xlabel('time waited (s)')
    ax.set_ylabel('count')
    plt.xlim(0, 300)
    ax.set_title(f'distribution of time waited, \n {file}')
    plt.show()
    fig.clear()
    print('mean time waited: ', np.mean(time_waited))

    # distribution of inter-arrival times
    sorted_arrival_times = np.unique([passenger.arrival_time for passenger in merged])
    inter_arrival_times = [(sorted_arrival_times[i + 1] - sorted_arrival_times[i]).total_seconds()
                           for i in range(len(sorted_arrival_times) - 1)]
    # limit to 60 seconds
    inter_arrival_times = [60 if t > 60 else t for t in inter_arrival_times]

    fig, ax = plt.subplots()
    ax.hist(inter_arrival_times, bins=60)
    ax.set_xlabel('inter arrival time (s)')
    ax.set_ylabel('count')
    ax.set_title(f'distribution of inter arrival times,\n {file}')
    plt.show()
    fig.clear()

    # distribution of number of passengers in elevator
    elev_occupants_history = [elev_occupants_history[f'       Lift {i}'] for i in range(1, 7)]
    fig, ax = plt.subplots()
    sns.violinplot(data=elev_occupants_history, ax=ax)
    ax.set_xticklabels([f'Lift {i}' for i in range(1, 7)])
    ax.set_ylabel('number of passengers')
    ax.set_title(f'distribution of number of passengers in elevator, \n {file}')
    plt.show()

    destinations = defaultdict(list)  # list of destinations for each hour

    for passenger in matched_passengers:
        hour = passenger.arrival_time.hour
        destinations[hour].append(passenger.destination)

    # distribution of destinations for hour 15.00
    fig, ax = plt.subplots()
    morning = destinations[6] + destinations[7] + destinations[8] + destinations[9] + destinations[10]
    afternoon = destinations[11] + destinations[12] + destinations[13] + destinations[14] + destinations[15]
    evening = destinations[16] + destinations[17] + destinations[18] + destinations[19] + destinations[20]
    sns.set_theme()
    plt.hist([morning, afternoon, evening], label=['morning (06.00-10.59)', 'afternoon (11.00-15.59)',
                                                   'evening (16.00-20.59)'], density=True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f'distribution of destinations \n {file}')
    ax.set_xlabel('floor')
    ax.set_ylabel('count')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.show()
