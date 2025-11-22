#!/usr/bin/env python3
"""COZ Daily Brief Automation Runner - see plan for details"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from elle.coz.sync_manager import SyncManager
from elle.coz.intelligence import IntelligenceEngine

DEFAULT_CONFIG = {
    "coz_dir": "elle/coz",
    "output_dir": "elle/coz/briefs",
    "hourly_rate": 25.0,
    "use_refinement": True
}

class BriefRunner:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._ensure_directories()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return DEFAULT_CONFIG
    
    def _setup_logging(self) -> logging.Logger:
        log_dir = Path("elle/coz/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('BriefRunner')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_dir / 'brief_runner.log')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(fh)
        return logger
    
    def _ensure_directories(self):
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def run(self) -> bool:
        try:
            self.logger.info("Starting daily brief generation")
            
            sync = SyncManager(coz_dir=self.config['coz_dir'])
            sync.parse_all()
            
            intelligence = IntelligenceEngine(sync, logger=self.logger)
            brief = intelligence.generate_daily_brief(
                hourly_rate=self.config['hourly_rate'],
                use_refinement=self.config['use_refinement']
            )
            
            output_file = Path(self.config['output_dir']) / f"coz_brief_{brief['date']}.md"
            with open(output_file, 'w') as f:
                f.write(brief['summary'])
            
            self.logger.info(f"Brief saved to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed: {e}", exc_info=True)
            return False

if __name__ == '__main__':
    runner = BriefRunner()
    sys.exit(0 if runner.run() else 1)
