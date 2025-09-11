from pyfiglet import figlet_format
from rich.console import Console
from rich.panel import Panel
from rich.box import HORIZONTALS

console = Console()
ascii_logo = f"[bold cyan]{figlet_format('Lulala')}[/bold cyan]"
console.print(Panel(ascii_logo, title="Welcome", subtitle="v1.0", border_style="cyan", width=100, box=HORIZONTALS))
