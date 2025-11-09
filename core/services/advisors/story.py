
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
        # Calculate time window: since last SA session for this username
        prev_sa = SmartAnalysis.objects.filter(
            username=sa.username,
            id__lt=sa.id
        ).order_by('-id').first()

        # Set bounds: prev SA started -> current SA started
        sa_end_utc = timezone.make_aware(sa.started) if timezone.is_naive(sa.started) else sa.started
        if prev_sa:
            sa_start_utc = timezone.make_aware(prev_sa.started) if timezone.is_naive(
                prev_sa.started) else prev_sa.started
        else:
            sa_start_utc = sa_end_utc - timedelta(hours=12)

            # Fetch StockStory articles within the time window
            articles = self._fetch_stockstory_articles(sa_start_utc, sa_end_utc)
            
            if not articles:
                logger.info("No StockStory articles found in time window")
                return

            # Process each article through Gemini
            for url, title in articles:
                self.news_flash(sa, title, url)

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
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Look for links that might be articles (StockStory uses /story/news/ pattern)
                if '/story/news/' in href and len(text) > 20:
                    # Skip navigation/category links
                    if any(skip in href.lower() for skip in ['/authors/', '/exclusives', '/chart-of', '/categories']):
                        continue
                    
                    # Build full URL
                    if href.startswith('/'):
                        full_url = 'https://www.barchart.com' + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # Try to find the article's timestamp
                    article_time = self._extract_article_time(link, soup)
                    
                    # If still no time found, try finding nearby time elements
                    if article_time is None:
                        article_time = self._find_nearby_time(link, soup)
                    
                    # Only include articles with timestamps (ignore sponsored content without timestamps)
                    # and within our time window
                    if article_time:
                        # Ensure article_time is timezone-aware for comparison
                        if article_time.tzinfo is None:
                            article_time = article_time.replace(tzinfo=dt_timezone.utc)
                        else:
                            # Convert to UTC if needed
                            article_time = article_time.astimezone(dt_timezone.utc)
                        
                        # Convert start/end times to UTC for comparison
                        start_utc = start_time.astimezone(dt_timezone.utc) if start_time.tzinfo else start_time.replace(tzinfo=dt_timezone.utc)
                        end_utc = end_time.astimezone(dt_timezone.utc) if end_time.tzinfo else end_time.replace(tzinfo=dt_timezone.utc)
                        
                        if start_utc <= article_time <= end_utc:
                            articles.append((full_url, text))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_articles = []
            for url, title in articles:
                if url not in seen:
                    seen.add(url)
                    unique_articles.append((url, title))
            
            logger.info(f"Found {len(unique_articles)} StockStory articles in time window")
            return unique_articles
                
        except requests.RequestException as e:
            logger.error(f"Error fetching StockStory page: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scraping StockStory page: {e}", exc_info=True)
            return []

    def _extract_article_time(self, link_element, soup):
        """
        Extract timestamp from article link or nearby elements.
        Returns datetime object (timezone-aware) or None if not found.
        """
        # Try to find time element near the link
        parent = link_element.parent
        if parent:
            # Look for <time> elements
            time_elem = parent.find('time')
            if time_elem:
                # Try datetime attribute
                dt_str = time_elem.get('datetime')
                if dt_str:
                    try:
                        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                # Try text content (e.g., "2 hours ago", "1 hour ago", "Tue Nov 4, 11:40PM CST")
                time_text = time_elem.get_text(strip=True)
                parsed = self._parse_time_text(time_text)
                if parsed:
                    return parsed
            
            # Look for date/time in sibling elements (including those that come after the link)
            for sibling in parent.find_all(['span', 'div', 'time', 'p'], class_=re.compile(r'date|time|ago|published|meta', re.I)):
                text = sibling.get_text(strip=True)
                parsed = self._parse_time_text(text)
                if parsed:
                    return parsed
                
                dt_str = sibling.get('datetime')
                if dt_str:
                    try:
                        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    except:
                        pass
            
            # Look for text content that contains date patterns (e.g., "StockStory - Tue Nov 4, 11:40PM CST")
            parent_text = parent.get_text()
            if parent_text:
                # Try to find date patterns in the parent's text
                parsed = self._parse_time_text(parent_text)
                if parsed:
                    return parsed
        
        # Also check next sibling elements (timestamps often appear below the link)
        next_sibling = link_element.find_next_sibling()
        if next_sibling:
            text = next_sibling.get_text(strip=True)
            parsed = self._parse_time_text(text)
            if parsed:
                return parsed
        
        # Check all following siblings (timestamps might be a few elements away)
        for i, sibling in enumerate(link_element.find_next_siblings()):
            text = sibling.get_text(strip=True)
            if text:
                parsed = self._parse_time_text(text)
                if parsed:
                    return parsed
            # Limit search to avoid going too far (check first 5 siblings)
            if i >= 4:
                break
        
        return None

    def _find_nearby_time(self, link_element, soup):
        """
        Search for time elements near the link in the DOM.
        """
        # Look in the same container/article element
        container = link_element.find_parent(['article', 'div', 'li'])
        if container:
            # Look for any time-related elements
            time_elems = container.find_all(['time', 'span', 'div', 'p'], 
                                           class_=re.compile(r'date|time|ago|published|meta', re.I))
            for elem in time_elems:
                # Try datetime attribute
                dt_str = elem.get('datetime')
                if dt_str:
                    try:
                        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                # Try text content
                text = elem.get_text(strip=True)
                parsed = self._parse_time_text(text)
                if parsed:
                    return parsed
            
            # Also check container text for date patterns
            container_text = container.get_text()
            if container_text:
                parsed = self._parse_time_text(container_text)
                if parsed:
                    return parsed
        
        return None

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
        now = datetime.now(dt_timezone.utc)
        
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
                    return func(match)
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
                logger.debug(f"Error parsing absolute time '{time_text}': {e}")
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


register(name="StockStory", python_class="Story")