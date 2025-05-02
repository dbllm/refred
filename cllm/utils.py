import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import re
import argparse
import yaml

def args_to_dict(args: argparse.Namespace):
  args_dict = vars(args)  # Convert Namespace to dict
  processed_dict = {}

  for key, value in args_dict.items():
    if key.endswith('_config'):
      # Load YAML content and add to the dict with a new key
      new_key = key[:-7]  # Remove '_config' suffix
      with open(value, 'r') as yaml_file:
        processed_dict[new_key] = yaml.safe_load(yaml_file)
    else:
      processed_dict[key] = value

  return processed_dict
  

def backup_existing_file(file_name):
  file_folder = os.path.dirname(file_name)
  if not Path(file_folder).exists():
    return
  base_name = os.path.basename(file_name)
  rename_patern = '{}[.]old[.](?P<number>\d*)'.format(base_name)
  
  numbers = [0]
  for name in os.listdir(file_folder):
    _ret = re.search(rename_patern, name)
    if _ret is not None:
      numbers.append(int(_ret.group('number')))
  
  next_id = max(numbers) + 1
  os.rename(file_name, '{}.old.{}'.format(file_name, next_id))
  # logging.info('renaming file [%s] to [%s]', file_name, next_id)
  print('renaming file', file_name, next_id)

def setup_logging(level=logging.DEBUG, to_file=True, log_name=None):
  prog_name = Path(sys.argv[0]).name
  if to_file:
    file_name = './log/auto_logs/{}/{}.log'.format(prog_name, datetime.now().isoformat())
    if log_name is not None:
      file_name = './log/named_logs/{}'.format(log_name)
      # rename the existing file is 
      if Path(file_name).exists():
        backup_existing_file(file_name)
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    # we do not append on existing logs.
    assert not Path(file_name).exists()
    logging.basicConfig(
      level=level, filename=file_name, 
      format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )
    print('logging to file, path:', file_name)
  else:
    logging.basicConfig(
      level=level, 
      format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )
    print('logging to console')
  logging.info('Executed command: {}'.format(' '.join(sys.argv)))