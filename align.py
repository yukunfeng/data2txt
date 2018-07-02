#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Align events in matches with comments
"""

import os
import re
from bs4 import BeautifulSoup


class Aligner(object):
    """Aligner class."""
    def __init__(self, players_info_path, f13m_dir, f24_dir):
        """Init Aligner. Note f13m and f24 file' names are
        conresponding to each other"""

        self.players_info_path = os.path.expanduser(players_info_path)
        self.f13m_dir = os.path.expanduser(f13m_dir)
        self.f24_dir = os.path.expanduser(f24_dir)

        # ID to player and team names
        self.id_name_map = self.load_names(self.players_info_path)
        # Contain player and team names
        self.player_names = self.id_name_map.values()

    def align_matches(self):
        """Align all matches specified in f13m_dir"""

        f13m_files = os.listdir(self.f13m_dir)
        for f13m_file in f13m_files:
            # Only process files in xml format
            if f13m_file.find('xml') < 0:
                continue
            f13m_file_path = os.path.join(self.f13m_dir, f13m_file)
            f24_file_path = re.sub('F13', 'F24', f13m_file_path)
            self.align_match(f13m_file_path, f24_file_path)

    @staticmethod
    def align_match(f13m_file_path, f24_file_path):
        """Align for one match"""
        time_events_map = Aligner.load_f24(f24_file_path)
        time_messages_map = Aligner.load_f13m(f13m_file_path)
        for time, messages in time_messages_map.items():
            events = time_events_map[time]
            print("for time {}, we have {} events".format(time, len(events)))

    @staticmethod
    def load_f13m(f13m_file_path):
        """Load f13m file"""

        with open(f13m_file_path) as file_pointer:
            f13m_soup = BeautifulSoup(file_pointer, "lxml")
            messages = f13m_soup('message')
            time_messages_map = {}

            for message in messages:
                time = message['time']
                # Pass message beyond match
                if time == '':
                    continue
                message_min = message['minute']
                if message_min not in time_messages_map:
                    time_messages_map[message_min] = []

                time_messages_map[message_min].append(message)

            return time_messages_map

    @staticmethod
    def load_f24(f24_file_path):
        """ Load f24 events for one match"""

        with open(f24_file_path) as file_pointer:
            f24_soup = BeautifulSoup(file_pointer, "lxml")
            events = f24_soup('event')

            time_events_map = {}
            for event in events:
                event_min = event['min']
                event_type = event['type_id']
                # Start, end events are excluded
                if event_type in ['34', '37']:
                    continue
                if event_min not in time_events_map:
                    time_events_map[event_min] = []

                time_events_map[event_min].append(event)

            return time_events_map

    @staticmethod
    def load_names(file_path):
        """Load player id, team id and their names

        :file_path: str
        :returns: dict

        """
        id_name_map = {}

        with open(file_path) as file_pointer:
            f42_soup = BeautifulSoup(file_pointer, "lxml")

        players = f42_soup('player')
        for player in players:
            uid = player['uid']
            name = player('name')[0].text
            id_name_map[uid] = name

        teams = f42_soup('team')
        for team in teams:
            uid = team['uid']
            name = team('name')[0].text
            id_name_map[uid] = name

        return id_name_map


if __name__ == "__main__":
    # Unit test
    f13m_file_path = "../mitools/opta_merge/F13M_gameid855255.xml" 
    f24_file_path = "../mitools/opta_merge/F24_gameid803173.xml"
    #  Aligner.load_f13m(f13m_file_path)
    #  Aligner.load_f24(f24_file_path)
    Aligner.align_match(f13m_file_path, f24_file_path)
