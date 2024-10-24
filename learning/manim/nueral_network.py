from manim import *

class NeuralNetwork(Scene):
    def construct(self):
        # Create layers
        input_layer = self.create_layer(3, 0, 0)
        hidden_layer = self.create_layer(4, 2, 0)
        output_layer = self.create_layer(2, 4, 0)

        # Create arrows between layers
        input_to_hidden_arrows = self.create_arrows(input_layer, hidden_layer)
        hidden_to_output_arrows = self.create_arrows(hidden_layer, output_layer)

        # Display the layers and connections
        self.play(LaggedStart(FadeIn(input_layer), FadeIn(hidden_layer), FadeIn(output_layer), lag_ratio=0.5))
        self.play(LaggedStart(*input_to_hidden_arrows, lag_ratio=0.1))
        self.play(LaggedStart(*hidden_to_output_arrows, lag_ratio=0.1))

        # Add labels
        self.add_labels(input_layer, "Input Layer")
        self.add_labels(hidden_layer, "Hidden Layer")
        self.add_labels(output_layer, "Output Layer")

        self.wait(2)

    def create_layer(self, num_nodes, x_shift, y_shift):
        """Create a layer of nodes."""
        layer = VGroup()
        for i in range(num_nodes):
            node = Circle(radius=0.2, color=BLUE)
            node.move_to(RIGHT * x_shift + UP * (i - num_nodes / 2) * 0.7 + DOWN * y_shift)
            layer.add(node)
        return layer

    def create_arrows(self, layer_a, layer_b):
        """Create arrows from layer_a to layer_b."""
        arrows = []
        for node_a in layer_a:
            for node_b in layer_b:
                arrow = Arrow(start=node_a.get_center(), end=node_b.get_center(), buff=0.1)
                arrows.append(arrow)
        return arrows

    def add_labels(self, layer, label_text):
        """Add a label above a layer."""
        label = Text(label_text, font_size=24)
        label.move_to(layer.get_center() + UP * 0.5)
        self.play(Write(label))
        self.wait(1)
