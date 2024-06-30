def analyze_python_code(code):
    tree = ast.parse(code)
    variable_names = set()
    variable_values = []
    strings = set()
    comments = set()

    class CodeAnalyzer(ast.NodeVisitor):
        
        def visit_Name(self, node):

            if node.id not in builtin_functions:
                variable_names.add(node.id)
            self.generic_visit(node)
        
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                strings.add(node.value)
            self.generic_visit(node)

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variable_names.add(target.id)
            if isinstance(node.value, ast.List):
                list_values = []
                for elem in node.value.elts:
                    if isinstance(elem, ast.Constant):
                        list_values.append(elem.value)
                variable_values.append(list_values)
            elif isinstance(node.value, ast.Constant):
                variable_values.append([node.value.value])
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            variable_names.add(node.name)
            for arg in node.args.args:
                variable_names.add(arg.arg)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            variable_names.add(node.name)
            self.generic_visit(node)

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    comments.update(re.findall(r'#.*', code))

    return {
        'variable_names': list(variable_names),
        'variable_values': variable_values,
        'strings': list(strings),
        'comments': list(comments)
    }

def mask_code(code, mask_ratio=0.5):
    code_lines = code.split('\n')[len(code.split('\n'))//4:]
    result_dict = analyze_python_code(code)
    
    variable_set = set(result_dict['variable_names'] + result_dict['variable_values'])
    string_comment_words = set(word for word in ' '.join(result_dict['strings'] + result_dict['comments']).split() if len(word) > 3)
    
    masked_lines = []
    masked_values = []
    
    for line in code_lines:
        tokens = re.split(r'(\W+)', line)
        masked_line = []
        
        for token in tokens:
            if random.random() < mask_ratio and (token in variable_set or token in string_comment_words):
                masked_values.append(token)
                masked_line.append('<mask>')
            else:
                masked_line.append(token)
        
        masked_lines.append(''.join(masked_line))
    
    return masked_lines, masked_values