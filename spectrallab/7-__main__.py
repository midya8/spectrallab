import argparse
import json
from pathlib import Path
from .report import build, audit, export


def main():
    p = argparse.ArgumentParser(description='Original synthetic graph clustering and numerical witnesses')
    sub = p.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit'); fit.add_argument('--output', required=True)
    check = sub.add_parser('audit'); check.add_argument('report')
    web = sub.add_parser('export'); web.add_argument('report'); web.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        if args.command == 'fit':
            report = build()
            Path(args.output).write_text(json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n')
            print(json.dumps(audit(report)))
        else:
            def pairs(items):
                result = {}
                for k, v in items:
                    if k in result: raise ValueError('duplicate key')
                    result[k] = v
                return result
            report = json.loads(Path(args.report).read_text(), object_pairs_hook=pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if args.command == 'audit': print(json.dumps(audit(report)))
            else: export(report, args.output); print('Exported checked dataset scripts')
    except (ValueError, KeyError, TypeError, OSError) as e:
        p.exit(2, f'Invalid report: {e}\n')

if __name__ == '__main__': main()
