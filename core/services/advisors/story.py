
import logging
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone as dt_timezone
from core.models import SmartAnalysis
from core.services.advisors.advisor import AdvisorBase, register
from django.utils import timezone

logger = logging.getLogger(__name__)

class Story(AdvisorBase):
    def discover(self, sa):
        """Discover stocks from StockStory news articles"""
        try:
            # Get time window: from previous SA to current SA
            prev_sa = SmartAnalysis.objects.filter(
                id__lt=sa.id
            ).order_by('-id').first()
            
            # Set bounds: prev SA started -> now (to catch articles published since SA started)
            end_time = timezone.now()
            if prev_sa:
                start_time = timezone.make_aware(prev_sa.started) if timezone.is_naive(prev_sa.started) else prev_sa.started
            else:
                # Fallback: last 24 hours if no previous SA
                start_time = end_time - timedelta(hours=24)
            
            # Fetch StockStory articles within the time window
            articles = self._fetch_stockstory_articles(start_time, end_time)
            
            if not articles:
                return
            
            # Track statistics
            total_found = len(articles)
            accepted_count = 0
            rejected_count = 0
            
            # Process each article through Gemini
            for article_data in articles:
                if len(article_data) == 3:
                    url, title, ticker_price_text = article_data
                else:
                    # Backward compatibility
                    url, title = article_data
                    ticker_price_text = ""
                
                # Filter: only accept articles about a single stock
                if self._is_single_stock_article(ticker_price_text):
                    self.news_flash(sa, title, url)
                    accepted_count += 1
                else:
                    rejected_count += 1
            
            # Summary statistics
            logger.info(f"StockStory: Summary - {total_found} articles found, {accepted_count} accepted, {rejected_count} rejected")

        except Exception as e:
            logger.error(f"StockStory discovery error: {e}", exc_info=True)

    def _fetch_stockstory_articles(self, start_time, end_time):
        """
        Scrape StockStory articles within the specified time window.
        
        Args:
            start_time: Start datetime (timezone-aware)
            end_time: End datetime (timezone-aware)
        
        Returns:
            List of tuples: [(url, title), ...] for articles within the time window
        """
        url = "https://www.barchart.com/news/authors/285/stockstory"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all article links
            all_links = soup.find_all('a', href=True)
            articles = []
            
            # Convert times to UTC for comparison
            start_utc = start_time.astimezone(dt_timezone.utc) if start_time.tzinfo else start_time.replace(tzinfo=dt_timezone.utc)
            end_utc = end_time.astimezone(dt_timezone.utc) if end_time.tzinfo else end_time.replace(tzinfo=dt_timezone.utc)
            
            logger.info(f"StockStory: Time window - {start_utc} to {end_utc}")
            
            article_links_found = 0
            articles_with_time = 0
            articles_in_window = 0
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Look for article links
                if '/story/news/' in href and len(text) > 20:
                    # Skip navigation/category links
                    if any(skip in href.lower() for skip in ['/authors/', '/exclusives', '/chart-of', '/categories']):
                        continue
                    
                    article_links_found += 1
                    
                    # Build full URL
                    if href.startswith('/'):
                        full_url = 'https://www.barchart.com' + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # Try to extract article timestamp
                    article_time = self._extract_article_time(link)
                    
                    # Only include articles with timestamps within our window
                    if article_time:
                        articles_with_time += 1
                        # Ensure timezone-aware
                        if article_time.tzinfo is None:
                            article_time = article_time.replace(tzinfo=dt_timezone.utc)
                        else:
                            article_time = article_time.astimezone(dt_timezone.utc)
                        
                        if start_utc <= article_time <= end_utc:
                            articles_in_window += 1
                            # Extract ticker:price pattern from nearby HTML elements
                            ticker_price_text = self._extract_ticker_price(link)
                            articles.append((full_url, text, ticker_price_text))
                        elif articles_in_window == 0 and articles_with_time <= 3:
                            # Log first few articles outside window for diagnosis
                            # Also log what time text was found
                            time_text_sample = self._get_time_text_sample(link)
                            logger.info(f"StockStory: Article outside window - Time: {article_time}, Window: {start_utc} to {end_utc}, Time text: '{time_text_sample}', Title: {text[:80]}")
                    elif article_links_found <= 3:
                        # Log first few articles without timestamps for diagnosis
                        logger.info(f"StockStory: Article without timestamp - Title: {text[:80]}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_articles = []
            for article_data in articles:
                url = article_data[0]
                if url not in seen:
                    seen.add(url)
                    unique_articles.append(article_data)
            
            logger.info(f"StockStory: Found {article_links_found} article links, {articles_with_time} with timestamps, {articles_in_window} in window, {len(unique_articles)} unique")
            return unique_articles
                
        except requests.RequestException as e:
            logger.error(f"Error fetching StockStory page: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scraping StockStory page: {e}", exc_info=True)
            return []

    def _get_time_text_sample(self, link_element):
        """Get a sample of time-related text from the article for debugging"""
        parent = link_element.parent
        if not parent:
            return ""
        
        # Check parent and siblings for time-related text
        search_elements = [parent]
        search_elements.extend(parent.find_all(['span', 'div', 'time', 'p']))
        search_elements.extend(link_element.find_next_siblings()[:5])
        
        for elem in search_elements:
            text = elem.get_text(strip=True) if hasattr(elem, 'get_text') else str(elem)
            if text and any(word in text.lower() for word in ['ago', 'hour', 'minute', 'day', 'stockstory']):
                return text[:100]
        
        return ""
    
    def _extract_article_time(self, link_element):
        """
        Extract timestamp from article link or nearby elements.
        Returns datetime object (timezone-aware) or None if not found.
        """
        # Strategy: Look in parent, then siblings, then container
        parent = link_element.parent
        if not parent:
            return None
        
        # 1. Check for <time> element with datetime attribute
        time_elem = parent.find('time')
        if time_elem:
            dt_str = time_elem.get('datetime')
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    return dt if dt.tzinfo else dt.replace(tzinfo=dt_timezone.utc)
                except:
                    pass
        
        # 2. Check parent and siblings for time-related text
        search_elements = [parent]
        search_elements.extend(parent.find_all(['span', 'div', 'time', 'p']))
        search_elements.extend(link_element.find_next_siblings()[:5])
        
        for elem in search_elements:
            text = elem.get_text(strip=True) if hasattr(elem, 'get_text') else str(elem)
            if text:
                parsed = self._parse_time_text(text)
                if parsed:
                    return parsed
        
        # 3. Check container (article/div/li) for time elements
        container = link_element.find_parent(['article', 'div', 'li'])
        if container:
            time_elems = container.find_all(['time', 'span', 'div'], 
                                            class_=re.compile(r'date|time|ago|published|meta', re.I))
            for elem in time_elems:
                dt_str = elem.get('datetime')
                if dt_str:
                    try:
                        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        return dt if dt.tzinfo else dt.replace(tzinfo=dt_timezone.utc)
                    except:
                        pass
                
                text = elem.get_text(strip=True)
                parsed = self._parse_time_text(text)
                if parsed:
                    return parsed
        
        return None

    def _extract_ticker_price(self, link_element):
        """
        Extract ticker:price pattern from nearby HTML elements.
        Looks for patterns like "ITW : 242.53" or "BLBD:54.86" in the article container.
        Returns the text containing ticker:price patterns, or empty string.
        """
        # Find the article container (limit search to this specific article)
        container = link_element.find_parent(['article', 'div', 'li', 'section'])
        if not container:
            container = link_element.parent
            if not container:
                return ""
        
        # Get all text from this container (but limit to reasonable size to avoid combining multiple articles)
        container_text = container.get_text(separator=' ', strip=True)
        
        # Limit to first 500 chars to avoid pulling in next article
        # The ticker:price should be near the link, so it should be in the first part
        container_text = container_text[:500]
        
        # Look for ticker:price pattern - handle both "TICKER:PRICE" and "TICKER : PRICE"
        # Pattern matches: "BLBD:54.86", "ITW : 242.53", "BLBD : 54.86", etc.
        ticker_pattern = r'([A-Z]{1,5})\s*:\s*\d+\.\d+'
        
        matches = re.findall(ticker_pattern, container_text, re.IGNORECASE)
        if matches:
            # Return just the portion of text that contains ticker patterns
            # Find the position of the first ticker pattern
            match_obj = re.search(ticker_pattern, container_text, re.IGNORECASE)
            if match_obj:
                start_pos = max(0, match_obj.start() - 50)  # Get some context before
                end_pos = min(len(container_text), match_obj.end() + 200)  # Get context after
                extracted = container_text[start_pos:end_pos]
                return extracted
        
        return ""

    def _parse_time_text(self, time_text):
        """
        Parse time strings including:
        - Relative: "1 hour ago", "2 hours ago", "30 minutes ago"
        - Absolute: "Tue Nov 4, 11:40PM CST", "StockStory - Tue Nov 4, 11:40PM CST"
        Returns datetime object (timezone-aware) or None.
        """
        if not time_text:
            return None
        
        time_text = time_text.strip()
        time_text_lower = time_text.lower()
        now = timezone.now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt_timezone.utc)
        else:
            now = now.astimezone(dt_timezone.utc)
        
        # First try relative time patterns
        relative_patterns = [
            (r'(\d+)\s*(?:hour|hr|h)\s*ago', lambda m: now - timedelta(hours=int(m.group(1)))),
            (r'(\d+)\s*(?:minute|min|m)\s*ago', lambda m: now - timedelta(minutes=int(m.group(1)))),
            (r'(\d+)\s*(?:day|d)\s*ago', lambda m: now - timedelta(days=int(m.group(1)))),
            (r'just now|now', lambda m: now),
        ]
        
        for pattern, func in relative_patterns:
            match = re.search(pattern, time_text_lower)
            if match:
                try:
                    result = func(match)
                    logger.debug(f"StockStory: Parsed '{time_text}' -> {result} (pattern: {pattern})")
                    return result
                except:
                    pass
        
        # Try absolute date format: "Tue Nov 4, 11:40PM CST" or "StockStory - Tue Nov 4, 11:40PM CST"
        # Pattern: DayOfWeek Month Day, Hour:MinuteAM/PM Timezone
        absolute_pattern = r'([A-Za-z]{3})\s+([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{1,2}):(\d{2})(AM|PM)\s+([A-Z]{3,4})'
        match = re.search(absolute_pattern, time_text)
        if match:
            try:
                day_name, month_name, day, hour, minute, am_pm, tz_str = match.groups()
                
                # Convert month name to number (case-insensitive)
                months = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month = months.get(month_name.lower(), 1)
                day = int(day)
                hour = int(hour)
                minute = int(minute)
                
                # Convert 12-hour to 24-hour
                if am_pm.upper() == 'PM' and hour != 12:
                    hour += 12
                elif am_pm.upper() == 'AM' and hour == 12:
                    hour = 0
                
                # Get current year (assume current year, or previous if date is in the future)
                current_year = datetime.now().year
                dt = datetime(current_year, month, day, hour, minute)
                
                # Convert timezone to UTC
                tz_offset = self._get_timezone_offset(tz_str)
                if tz_offset is not None:
                    # Create timezone-aware datetime
                    dt = dt.replace(tzinfo=dt_timezone(timedelta(hours=tz_offset)))
                    # Convert to UTC
                    dt = dt.astimezone(dt_timezone.utc)
                else:
                    # If timezone unknown, assume UTC
                    dt = dt.replace(tzinfo=dt_timezone.utc)
                
                # If the date is in the future (e.g., we're in December but the date is January),
                # it's probably from last year
                if dt > now:
                    dt = dt.replace(year=current_year - 1)
                
                return dt
            except Exception as e:
                pass
        
        return None

    def _get_timezone_offset(self, tz_str):
        """
        Convert timezone abbreviation to UTC offset in hours.
        Returns offset in hours or None if unknown.
        """
        # Common US timezones
        tz_map = {
            'EST': -5, 'EDT': -4,  # Eastern
            'CST': -6, 'CDT': -5,  # Central
            'MST': -7, 'MDT': -6,  # Mountain
            'PST': -8, 'PDT': -7,  # Pacific
            'AKST': -9, 'AKDT': -8,  # Alaska
            'HST': -10,  # Hawaii
        }
        
        return tz_map.get(tz_str.upper())

    def _is_single_stock_article(self, ticker_price_text):
        """
        Check if article relates to exactly one stock.
        Looks for patterns like "TICKER : PRICE" or "TICKER: PRICE" in the ticker_price_text.
        Returns True if exactly one stock ticker pattern is found.
        """
        if not ticker_price_text:
            return False
        
        # Pattern to match stock ticker with price: "TICKER : PRICE" or "TICKER:PRICE" (with or without space)
        # Matches: "BLBD : 54.86", "BLBD:54.86", "CECO : 51.44", "ASGN : 44.56", "ITW : 242.53", etc.
        # Case-insensitive to handle variations
        ticker_pattern = r'([A-Z]{1,5})\s*:\s*\d+\.\d+'
        
        matches = re.findall(ticker_pattern, ticker_price_text, re.IGNORECASE)
        
        # Must have exactly one unique ticker
        unique_tickers = set(matches)
        
        return len(unique_tickers) == 1


register(name="StockStory", python_class="Story")