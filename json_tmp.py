#!/usr/bin/env python3
import json

data = json.loads(open("./new_parallel_delta5_9.json.dev").read())
max_source = 0
max_quantifier = 0
spliter = "\t"
padding = "-100"
loop_field = 10
for count, item in enumerate(data, 1):
    sources = item["source"]
    if len(sources) > max_source:
        max_source = len(sources)
    for source in sources:
        details = source["details"]
        if len(details) > max_quantifier:
            max_quantifier = len(details)

#  print(f"max_source: {max_source}")
#  exit(0)

for count, item in enumerate(data, 1):
    target = item["target"]
    comment = target["comment"]
    if comment.find(spliter) >= 0:
        print(f"{comment} has {spliter}")
        exit(1)
    cmt_id = target["id"]
    out_line = f"{comment}{spliter}"

    sources = item["source"]
    out_line += f"{len(sources)}{spliter}"
    for i in range(max_source):
        if i > len(sources) - 1:
            padding_line = spliter.join([padding] * loop_field * (max_source - len(sources)))
            out_line = out_line.strip()
            out_line += f"{spliter}{padding_line}"
            #  print(f"{len(out_line.split(spliter))} {len(padding_line.split(spliter))}")
            break
        source = sources[i]
        event_id = source["event_id"]
        out_line += f"{event_id}{spliter}"
        type_id = source["type_id"]
        out_line += f"{type_id}{spliter}"
        minute = source["minute"]
        out_line += f"{minute}{spliter}"
        second = source["second"]
        out_line += f"{second}{spliter}"
        outcome = source["outcome"]
        out_line += f"{outcome}{spliter}"
        x = source["x"]
        out_line += f"{x}{spliter}"
        y = source["y"]
        out_line += f"{y}{spliter}"
        end_x = source["end_x"]
        out_line += f"{end_x}{spliter}"
        end_y = source["end_y"]
        out_line += f"{end_y}{spliter}"
        details = source["details"]
        quantifiers = []
        for detail in details:
            quantifiers.append(str(detail[-1]))

        #  quantifiers += [padding] * (max_quantifier - len(quantifiers))
        quantifiers_str = " ".join(quantifiers)
        if quantifiers_str == "":
            quantifiers_str = padding
        out_line += f"{quantifiers_str}{spliter}"
        num1 = len(out_line.split(spliter))
    out_line = out_line.strip()
    print(out_line)
    #  total = len(out_line.split("\t"))
    #  print(f"{total} = {num1} + {num2}")
