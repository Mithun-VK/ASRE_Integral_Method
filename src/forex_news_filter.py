"""
🌍 FOREX NEWS FILTER V1.1 - IMPROVED ANTI-DETECTION
=======================================================
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, time
import pytz
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional, Tuple
import time as time_module
import random

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


NEWS_FILTER_CONFIG = {
    'enabled': True,
    'buffer_before_minutes': 30,
    'buffer_after_minutes': 15,
    'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'NZD'],
    'timezone': 'US/Eastern',
    'cache_duration_minutes': 60,
    'retry_attempts': 2,
    'request_delay_seconds': 2,  # Delay between requests
}

HIGH_PRIORITY_EVENTS = [
    'Non-Farm', 'NFP', 'FOMC', 'Interest Rate', 'GDP', 'CPI', 'Inflation',
    'Unemployment', 'Retail Sales', 'Employment', 'Central Bank', 'ECB', 'BoE', 'BoJ'
]

# Rotate user agents to avoid detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
]


class ForexNewsFilter:
    """
    Forex Factory news filter with improved anti-detection and caching.
    """
    
    def __init__(self, 
                 timezone: str = None,
                 buffer_before: int = None,
                 buffer_after: int = None,
                 currencies: List[str] = None):
        self.timezone = pytz.timezone(timezone or NEWS_FILTER_CONFIG['timezone'])
        self.buffer_before = buffer_before or NEWS_FILTER_CONFIG['buffer_before_minutes']
        self.buffer_after = buffer_after or NEWS_FILTER_CONFIG['buffer_after_minutes']
        self.currencies = currencies or NEWS_FILTER_CONFIG['currencies']
        
        self.ff_url = "https://www.forexfactory.com/calendar"
        self.high_impact_events = []
        self.last_fetch = None
        self.fetch_error = None
        self.events_cache = {}  # Cache events by date
        self.last_request_time = None
        
        logger.info(f"📰 News Filter initialized: {self.buffer_before}min before, "
                   f"{self.buffer_after}min after (timezone: {self.timezone})")
    
    def _get_random_headers(self) -> Dict:
        """Get random headers to avoid detection."""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        if self.last_request_time:
            elapsed = time_module.time() - self.last_request_time
            wait_time = NEWS_FILTER_CONFIG['request_delay_seconds'] - elapsed
            if wait_time > 0:
                time_module.sleep(wait_time)
        self.last_request_time = time_module.time()
    
    def fetch_calendar(self, date: Optional[datetime] = None) -> List[Dict]:
        """
        Fetch calendar with caching and rate limiting.
        """
        target_date = date or datetime.now(self.timezone)
        date_key = target_date.date()
        
        # Check cache first
        if date_key in self.events_cache:
            cache_time, cached_events = self.events_cache[date_key]
            if (datetime.now(self.timezone) - cache_time).total_seconds() < NEWS_FILTER_CONFIG['cache_duration_minutes'] * 60:
                logger.debug(f"   Using cached events for {date_key}")
                self.high_impact_events = cached_events
                return cached_events
        
        # Try fetching with retries
        for attempt in range(NEWS_FILTER_CONFIG['retry_attempts']):
            try:
                self._rate_limit()  # Enforce rate limiting
                
                date_str = target_date.strftime('%b%d.%Y').lower()
                url = f"{self.ff_url}?day={date_str}"
                
                headers = self._get_random_headers()
                
                logger.debug(f"   Fetching FF calendar (attempt {attempt + 1}): {date_str}")
                
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                calendar_rows = soup.find_all('tr', class_='calendar__row')
                
                events = []
                current_date = target_date.date()
                
                for row in calendar_rows:
                    # Parse time
                    time_cell = row.find('td', class_='calendar__time')
                    if not time_cell:
                        continue
                    
                    event_time_str = time_cell.text.strip()
                    if not event_time_str or event_time_str in ['All Day', 'Tentative', 'Day']:
                        continue
                    
                    # Parse currency
                    currency_cell = row.find('td', class_='calendar__currency')
                    if not currency_cell:
                        continue
                    
                    currency = currency_cell.text.strip()
                    if currency not in self.currencies:
                        continue
                    
                    # Check impact (high only)
                    impact_cell = row.find('td', class_='calendar__impact')
                    if not impact_cell:
                        continue
                    
                    impact_span = impact_cell.find('span')
                    if not impact_span or 'icon--ff-impact-red' not in impact_span.get('class', []):
                        continue
                    
                    # Get title
                    event_cell = row.find('td', class_='calendar__event')
                    if not event_cell:
                        continue
                    
                    event_title = event_cell.text.strip()
                    
                    try:
                        event_datetime = self._parse_event_time(current_date, event_time_str)
                        if not event_datetime:
                            continue
                        
                        is_priority = any(keyword.lower() in event_title.lower() 
                                         for keyword in HIGH_PRIORITY_EVENTS)
                        
                        events.append({
                            'currency': currency,
                            'title': event_title,
                            'datetime': event_datetime,
                            'impact': 'High',
                            'priority': is_priority,
                            'time_str': event_time_str
                        })
                        
                    except Exception as e:
                        logger.debug(f"   Error parsing event: {e}")
                        continue
                
                # Cache successful fetch
                self.events_cache[date_key] = (datetime.now(self.timezone), events)
                self.high_impact_events = sorted(events, key=lambda x: x['datetime'])
                self.last_fetch = datetime.now(self.timezone)
                self.fetch_error = None
                
                if len(events) > 0:
                    logger.info(f"   ✅ Fetched {len(events)} high-impact events for {date_str}")
                else:
                    logger.info(f"   ℹ️  No high-impact events for {date_str}")
                
                return events
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    logger.warning(f"   ⚠️  Forex Factory blocked request (403) - attempt {attempt + 1}")
                    if attempt < NEWS_FILTER_CONFIG['retry_attempts'] - 1:
                        wait_time = random.uniform(3, 7)  # Random delay
                        logger.warning(f"   → Retrying in {wait_time:.1f}s...")
                        time_module.sleep(wait_time)
                    else:
                        logger.warning(f"   → All retry attempts failed. Continuing without news filter.")
                        self.fetch_error = "403 Forbidden - Forex Factory blocking requests"
                        return []
                else:
                    raise
                    
            except Exception as e:
                logger.warning(f"   ⚠️  Error fetching calendar: {e}")
                if attempt < NEWS_FILTER_CONFIG['retry_attempts'] - 1:
                    time_module.sleep(2)
                else:
                    self.fetch_error = str(e)
                    return []
        
        return []
    
    def _parse_event_time(self, date: datetime.date, time_str: str) -> Optional[datetime]:
        """Parse event time."""
        try:
            for fmt in ['%I:%M%p', '%H:%M']:
                try:
                    event_time = datetime.strptime(time_str, fmt).time()
                    event_datetime = datetime.combine(date, event_time)
                    event_datetime = self.timezone.localize(event_datetime)
                    return event_datetime
                except ValueError:
                    continue
            return None
        except Exception as e:
            logger.debug(f"   Error parsing time '{time_str}': {e}")
            return None
    
    def is_safe_to_trade(self, 
                         current_time: datetime,
                         pair: str = None) -> Tuple[bool, Optional[Dict]]:
        """
        Check if safe to trade (with graceful fallback on fetch errors).
        """
        # Convert to aware datetime
        if not isinstance(current_time, datetime):
            current_time = pd.Timestamp(current_time).to_pydatetime()
        
        if current_time.tzinfo is None:
            current_time = self.timezone.localize(current_time)
        else:
            current_time = current_time.astimezone(self.timezone)
        
        # Refresh if needed (but don't spam requests)
        current_date = current_time.date()
        if current_date not in self.events_cache:
            # Only fetch once per date
            self.fetch_calendar(current_time)
        
        # If fetch failed, assume safe (graceful degradation)
        if self.fetch_error:
            return True, None
        
        # Filter by pair currencies
        relevant_currencies = self._get_pair_currencies(pair) if pair else self.currencies
        
        # Check events
        for event in self.high_impact_events:
            if event['currency'] not in relevant_currencies:
                continue
            
            event_time = event['datetime']
            time_diff_minutes = (event_time - current_time).total_seconds() / 60
            
            if -self.buffer_after <= time_diff_minutes <= self.buffer_before:
                return False, {
                    'event': event['title'],
                    'currency': event['currency'],
                    'time': event_time.strftime('%H:%M'),
                    'minutes_until': int(time_diff_minutes),
                    'priority': event.get('priority', False)
                }
        
        return True, None
    
    def _get_pair_currencies(self, pair: str) -> List[str]:
        """Extract currencies from pair."""
        if '/' in pair:
            base, quote = pair.split('/')
            return [base.upper(), quote.upper()]
        return self.currencies
    
    def get_todays_events(self, date: Optional[datetime] = None) -> str:
        """Get formatted event list."""
        if not self.high_impact_events or date:
            self.fetch_calendar(date)
        
        if self.fetch_error:
            return f"   ⚠️  News filter unavailable: {self.fetch_error}\n   → Continue with manual calendar check!"
        
        if not self.high_impact_events:
            return "   ℹ️  No high-impact news scheduled"
        
        output = "\n📰 HIGH-IMPACT NEWS TODAY:\n"
        for event in self.high_impact_events:
            priority_flag = "⚠️ " if event.get('priority') else "   "
            output += (f"{priority_flag}{event['datetime'].strftime('%H:%M')} | "
                      f"{event['currency']:3s} | {event['title']}\n")
        
        return output.rstrip()


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*80)
    print("📰 FOREX NEWS FILTER - STANDALONE TEST V1.1")
    print("="*80)
    
    news_filter = ForexNewsFilter(timezone='US/Eastern')
    events = news_filter.fetch_calendar()
    
    print(news_filter.get_todays_events())
    
    # Test current time
    now = datetime.now(news_filter.timezone)
    
    for pair in ['USD/JPY', 'EUR/USD']:
        is_safe, next_event = news_filter.is_safe_to_trade(now, pair)
        
        if is_safe:
            print(f"\n   ✅ {pair}: SAFE TO TRADE")
        else:
            print(f"\n   ⚠️  {pair}: DO NOT TRADE!")
            print(f"      {next_event['event']} in {next_event['minutes_until']:+d} min")
    
    print("\n" + "="*80 + "\n")
