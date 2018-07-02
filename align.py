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


def load_key_value_from_file(file_path):
    """Load pairs from document of opta. This is for debug

    :file_path: str
    :returns: dict

    """

    key_values = {}
    with open(file_path, 'r') as file_pointer:
        for line in file_pointer:
            line = line.strip()
            # Skip efile_pointer
            if line == "":
                continue

            tokens = line.split('\t')
            if len(tokens) < 2:
                continue

            key = tokens[0]
            value1 = tokens[1:]
            key_values[key] = value1

    return key_values


class Aligner(object):
    """Aligner class."""
    def __init__(self, debug=False):
        """Init Aligner.
        """

        # Load resources for debuging
        self.debug = debug
        if debug:
            self.period_key_values = load_key_value_from_file(
                "./appendix10_period.txt"
            )
            self.event_key_values = load_key_value_from_file(
                "./appendix1_event_type.txt"
            )
            self.qualifier_key_values = load_key_value_from_file(
                "./appendix2_qualifier_type.txt"
            )
            self.f24_field_key_values = load_key_value_from_file(
                "./f24_field_meaning.txt"
            )
            self.outcome_key_values = load_key_value_from_file(
                "./appendix8_outcome.txt"
            )
            self.id_name_map = load_names(
                "./F42_competiiton8_seasonid2016.xml"
            )

    def add_debug_to_event(self, events, f13_path):
        """Add debug information to events
        events: soup type
        f13_path: the coressponding f13path
        """

        with open(f13_path) as file_pointer:
            f13_soup = BeautifulSoup(file_pointer, "lxml")
        for event in events:
            # Add qualifier information
            qualifiers = event('q')
            for qualifier in qualifiers:
                key = qualifier['qualifier_id']
                meaning = "{}(NotFound)".format(key)
                if key in self.qualifier_key_values:
                    meaning = self.qualifier_key_values[key]
                    #  meaning = "{}({})".format(key, meaning)
                qualifier['qualifier_id'] = meaning[0]

            # Add outcome
            #  try:
                #  meaning = self.outcome_key_values[event['type_id']]
                #  event['outcome'] += "({})".format(meaning)
            #  except Exception as e:
                #  pass

            # Add explaination to type id
            key = event['type_id']
            meaning = "{}(NotFound)".format(key)
            if key in self.event_key_values:
                meaning = self.event_key_values[key]
                #  meaning = "{}({})".format(key, meaning)
            event['type_id'] = meaning[0]

            # Add period to period id
            key = event['period_id']
            meaning = "{}(NotFound)".format(key)
            if key in self.period_key_values:
                meaning = self.period_key_values[key]
                #  meaning = "{}({})".format(key, meaning)
            event['period_id'] = meaning[0]

            # Add assist and keypass
            try:
                meaning = self.f24_field_key_values['assist']
                event['assist'] += "({})".format(meaning)
            except Exception as e:
                pass
            try:
                meaning = self.f24_field_key_values['keypass']
                event['keypass'] += "({})".format(meaning)
            except Exception as e:
                pass

            # Add names to id
            try:
                team_name = self.id_name_map['t' + event['team_id']]
                player_name = self.id_name_map['p' + event['player_id']]
                event['team_id'] = team_name
                event['player_id'] = player_name
            except Exception as e:
                pass

            # Add f13 comments to event
            event_id = event['id']
            message = f13_soup('message', {'id': event_id})
            # F24's event doesn't contain any F13 comments
            if len(message) == 0:
                continue
            comment = message[0]['comment']
            f13_type = message[0]['type']
            event['f13_comment'] = comment
            event['f13_type'] = f13_type

    def align_matches(self):
        """Align all matches specified in f13m_dir"""
        pass
        #  f13m_files = os.listdir(self.f13m_dir)
        #  for f13m_file in f13m_files:
            # Only process files in xml format
            #  if f13m_file.find('xml') < 0:
                #  continue
            #  f13m_file_path = os.path.join(self.f13m_dir, f13m_file)
            #  f24_file_path = re.sub('F13', 'F24', f13m_file_path)
            #  self.align_match(f13m_file_path, f24_file_path)

    def event_to_string(self, event):
        """convert event to string for debug"""
        attributes = []
        keys = [
            'min',
            'sec',
            'outcome',
            'period_id',
            'player_id',
            'team_id',
            'type_id',
            'x',
            'y'
        ]
        for key in keys:
            try:
                val = event[key]
            except Exception as e:
                val = "NON"
            attributes.append(val)
        string = "|".join(attributes)
        return string

    def align_match(self, f13m_file_path, f24_file_path):
        """Align for one match"""
        time_events_map = load_f24(f24_file_path)
        time_messages_map = load_f13m(f13m_file_path)
        f13_file_path = re.sub('F13M', 'F13', f13m_file_path)
        for time, messages in time_messages_map.items():
            events = time_events_map[time]
            if self.debug:
                self.add_debug_to_event(events, f13_file_path)
                for event in events:
                    string = self.event_to_string(event)
                    print(string)
                for message in messages:
                    print(message['comment'])


if __name__ == "__main__":
    # Unit test
    f13m_file_path = "../mitools/opta_merge/F13M_gameid855255.xml" 
    f24_file_path = "../mitools/opta_merge/F24_gameid803173.xml"
    #  load_f13m(f13m_file_path)
    #  load_f24(f24_file_path)
    aligner = Aligner(debug=True)
    aligner.align_match(f13m_file_path, f24_file_path)
    #  aligner = Align()
