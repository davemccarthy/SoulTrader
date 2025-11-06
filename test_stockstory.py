import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import re

def get_stockstory_articles(hours=1):
    """
    Scrape StockStory articles from the last N hours.
    
    Args:
        hours: Number of hours to look back (default: 1)
    
    Returns:
        List of tuples: [(url, title), ...] for articles from the last N hours
    """
    url = "https://www.barchart.com/news/authors/285/stockstory"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Calculate cutoff time (N hours ago)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
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
                article_time = _extract_article_time(link, soup, cutoff_time)
                
                # If still no time found, try finding nearby time elements
                if article_time is None:
                    article_time = _find_nearby_time(link, soup, cutoff_time)
                
                # Only include articles with timestamps (ignore sponsored content without timestamps)
                if article_time and article_time >= cutoff_time:
                    articles.append((full_url, text))
                # If no timestamp found, skip this article (likely sponsored content)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_articles = []
        for url, title in articles:
            if url not in seen:
                seen.add(url)
                unique_articles.append((url, title))
        
        return unique_articles
            
    except Exception as e:
        print(f"Error scraping StockStory page: {e}")
        return []


def _extract_article_time(link_element, soup, cutoff_time):
    """
    Extract timestamp from article link or nearby elements.
    Returns datetime object or None if not found.
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
            parsed = _parse_time_text(time_text, cutoff_time)
            if parsed:
                return parsed
        
        # Look for date/time in sibling elements (including those that come after the link)
        for sibling in parent.find_all(['span', 'div', 'time', 'p'], class_=re.compile(r'date|time|ago|published|meta', re.I)):
            text = sibling.get_text(strip=True)
            parsed = _parse_time_text(text, cutoff_time)
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
            parsed = _parse_time_text(parent_text, cutoff_time)
            if parsed:
                return parsed
    
    # Also check next sibling elements (timestamps often appear below the link)
    next_sibling = link_element.find_next_sibling()
    if next_sibling:
        text = next_sibling.get_text(strip=True)
        parsed = _parse_time_text(text, cutoff_time)
        if parsed:
            return parsed
    
    # Check all following siblings (timestamps might be a few elements away)
    for i, sibling in enumerate(link_element.find_next_siblings()):
        text = sibling.get_text(strip=True)
        if text:
            parsed = _parse_time_text(text, cutoff_time)
            if parsed:
                return parsed
        # Limit search to avoid going too far (check first 5 siblings)
        if i >= 4:
            break
    
    return None


def _find_nearby_time(link_element, soup, cutoff_time):
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
            parsed = _parse_time_text(text, cutoff_time)
            if parsed:
                return parsed
        
        # Also check container text for date patterns
        container_text = container.get_text()
        if container_text:
            parsed = _parse_time_text(container_text, cutoff_time)
            if parsed:
                return parsed
    
    return None


def _parse_time_text(time_text, cutoff_time):
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
    now = datetime.now(timezone.utc)
    
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
            tz_offset = _get_timezone_offset(tz_str)
            if tz_offset is not None:
                # Create timezone-aware datetime
                dt = dt.replace(tzinfo=timezone(timedelta(hours=tz_offset)))
                # Convert to UTC
                dt = dt.astimezone(timezone.utc)
            else:
                # If timezone unknown, assume UTC
                dt = dt.replace(tzinfo=timezone.utc)
            
            # If the date is in the future (e.g., we're in December but the date is January),
            # it's probably from last year
            if dt > now:
                dt = dt.replace(year=current_year - 1)
            
            return dt
        except Exception as e:
            pass
    
    return None


def _get_timezone_offset(tz_str):
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


def get_latest_stockstory_url():
    """Backward compatibility: returns latest article URL and title"""
    articles = get_stockstory_articles(hours=24)  # Get articles from last 24 hours
    if articles:
        return articles[0]  # Return (url, title) tuple
    return None, None


if __name__ == "__main__":
    # Get articles from last 12 hours (as a test)
    articles = get_stockstory_articles(hours=12)
    
    if articles:
        print(f"Found {len(articles)} article(s) from the last 12 hours:\n")
        for i, (url, title) in enumerate(articles, 1):
            print(f"{i}. {title}")
            print(f"   {url}\n")
    else:
        print("No articles found from the last 12 hours")
