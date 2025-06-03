class HTMLResponseGenerator:
    def __init__(self):
        self.disclaimer = """
        <div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 my-6 rounded-md text-sm">
            <strong class="text-red-600">⚠️ This information is for educational purposes only and should not be considered medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.</strong>
        </div>
        """
        self.title = "Medical Information"
        self.sections = {}

    def set_data(self, sections: dict, title: str = "Medical Information"):
        self.sections = sections
        self.title = title

    def _build_sections(self) -> str:
        html = ""
        for title, content in self.sections.items():
            html += f'<h2 class="text-xl font-bold text-blue-700 mt-8 mb-2 border-b-2 border-blue-200 pb-1">{title}</h2>\n'
            html += '<div class="bg-blue-50 border-l-4 border-blue-300 p-4 rounded-md mb-6">\n'
            if isinstance(content, list):
                html += '<ul class="list-disc list-inside space-y-1">\n'
                for item in content:
                    html += f"<li>{item}</li>\n"
                html += "</ul>\n"
            elif isinstance(content, str):
                html += f"<p>{content}</p>\n"
            html += "</div>\n"
        return html

    def generate_html(self) -> str:
        """Generate Tailwind-styled HTML output from internal state."""
        return f"""
            <main class="max-w-3xl mx-auto p-6">
                <section class="bg-white p-6 rounded-xl shadow-md">
                    {self.disclaimer}
                    <h1 class="text-3xl font-bold text-blue-800 mb-4">{self.title}</h1>
                    {self._build_sections()}
                    {self.disclaimer}
                </section>
            </main>
        """
