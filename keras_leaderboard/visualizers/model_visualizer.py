"""
Keras Leaderboard - Model Visualizer.

This module provides tools for advanced visualization of Keras model architectures
and training metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.utils import plot_model


class ModelVisualizer:
    """Visualize Keras model architectures and training metrics."""
    
    def __init__(self, output_dir=None):
        """Initialize the model visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations.
                If None, visualizations will not be saved.
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up consistent styling
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_model_architecture(self, model, model_name, to_file=None, show_shapes=True, 
                               show_layer_names=True, rankdir='TB'):
        """Generate a visual representation of the model architecture.
        
        Args:
            model: Keras model to visualize
            model_name (str): Name of the model
            to_file (str, optional): Path to save the visualization
            show_shapes (bool): Whether to display shape information
            show_layer_names (bool): Whether to show layer names
            rankdir (str): Direction of graph layout ('TB' - top to bottom,
                'LR' - left to right)
        
        Returns:
            str: Path to the generated visualization file
        """
        if to_file is None and self.output_dir:
            to_file = os.path.join(self.output_dir, f"{model_name}_architecture.png")
        
        try:
            plot_model(model, to_file=to_file, show_shapes=show_shapes, 
                      show_layer_names=show_layer_names, rankdir=rankdir)
            print(f"Model architecture saved to {to_file}")
            return to_file
        except Exception as e:
            print(f"Error plotting model architecture: {e}")
            return None
    
    def plot_training_history(self, history, model_name, metrics=None, use_plotly=False):
        """Plot the training history metrics.
        
        Args:
            history: Keras history object or dictionary
            model_name (str): Name of the model
            metrics (list, optional): List of metrics to plot.
                If None, all available metrics will be plotted.
            use_plotly (bool): Whether to use Plotly for interactive plots
        
        Returns:
            dict: Paths to the generated visualization files
        """
        # Convert history to dict if it's a History object
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        # Determine metrics to plot
        if metrics is None:
            metrics = [m for m in history_dict.keys() if not m.startswith('val_')]
        
        output_files = {}
        
        # Create directory for plots
        if self.output_dir:
            plots_dir = os.path.join(self.output_dir, f"{model_name}_plots")
            os.makedirs(plots_dir, exist_ok=True)
        else:
            plots_dir = None
        
        # Plot each metric
        for metric in metrics:
            val_metric = f'val_{metric}' if f'val_{metric}' in history_dict else None
            
            if use_plotly:
                fig = self._plotly_history_plot(history_dict, metric, val_metric, model_name)
                
                if plots_dir:
                    plot_path = os.path.join(plots_dir, f"{model_name}_{metric}.html")
                    fig.write_html(plot_path)
                    output_files[metric] = plot_path
            else:
                fig = self._matplotlib_history_plot(history_dict, metric, val_metric, model_name)
                
                if plots_dir:
                    plot_path = os.path.join(plots_dir, f"{model_name}_{metric}.png")
                    fig.savefig(plot_path, bbox_inches='tight')
                    plt.close(fig)
                    output_files[metric] = plot_path
        
        return output_files
    
    def _matplotlib_history_plot(self, history_dict, metric, val_metric, model_name):
        """Create a matplotlib plot for training history.
        
        Args:
            history_dict (dict): Training history dictionary
            metric (str): Metric to plot
            val_metric (str): Validation metric to plot
            model_name (str): Name of the model
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots()
        epochs = range(1, len(history_dict[metric]) + 1)
        
        ax.plot(epochs, history_dict[metric], 'b-', label=f'Training {metric}')
        if val_metric:
            ax.plot(epochs, history_dict[val_metric], 'r-', label=f'Validation {metric}')
        
        ax.set_title(f'{model_name}: {metric.capitalize()} over epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def _plotly_history_plot(self, history_dict, metric, val_metric, model_name):
        """Create a Plotly plot for training history.
        
        Args:
            history_dict (dict): Training history dictionary
            metric (str): Metric to plot
            val_metric (str): Validation metric to plot
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        epochs = list(range(1, len(history_dict[metric]) + 1))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=history_dict[metric],
            mode='lines+markers',
            name=f'Training {metric}',
            line=dict(color='blue')
        ))
        
        if val_metric:
            fig.add_trace(go.Scatter(
                x=epochs, 
                y=history_dict[val_metric],
                mode='lines+markers',
                name=f'Validation {metric}',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title=f'{model_name}: {metric.capitalize()} over epochs',
            xaxis_title='Epochs',
            yaxis_title=metric.capitalize(),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def plot_model_comparison(self, models_data, metric='val_accuracy', use_plotly=False):
        """Generate a comparison plot of multiple models.
        
        Args:
            models_data (dict): Dictionary mapping model names to their history dicts
            metric (str): Metric to use for comparison
            use_plotly (bool): Whether to use Plotly for interactive plots
            
        Returns:
            str: Path to the generated comparison plot
        """
        if use_plotly:
            fig = self._plotly_comparison_plot(models_data, metric)
            
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"model_comparison_{metric}.html")
                fig.write_html(plot_path)
                return plot_path
        else:
            fig = self._matplotlib_comparison_plot(models_data, metric)
            
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"model_comparison_{metric}.png")
                fig.savefig(plot_path, bbox_inches='tight')
                plt.close(fig)
                return plot_path
    
    def _matplotlib_comparison_plot(self, models_data, metric):
        """Create a matplotlib comparison plot.
        
        Args:
            models_data (dict): Dictionary mapping model names to their history dicts
            metric (str): Metric to use for comparison
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots()
        
        for model_name, history_dict in models_data.items():
            if hasattr(history_dict, 'history'):
                history_dict = history_dict.history
                
            if metric in history_dict:
                epochs = range(1, len(history_dict[metric]) + 1)
                ax.plot(epochs, history_dict[metric], '-o', label=model_name)
        
        ax.set_title(f'Model Comparison: {metric.capitalize()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def _plotly_comparison_plot(self, models_data, metric):
        """Create a Plotly comparison plot.
        
        Args:
            models_data (dict): Dictionary mapping model names to their history dicts
            metric (str): Metric to use for comparison
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        fig = go.Figure()
        
        for model_name, history_dict in models_data.items():
            if hasattr(history_dict, 'history'):
                history_dict = history_dict.history
                
            if metric in history_dict:
                epochs = list(range(1, len(history_dict[metric]) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs, 
                    y=history_dict[metric],
                    mode='lines+markers',
                    name=model_name
                ))
        
        fig.update_layout(
            title=f'Model Comparison: {metric.capitalize()}',
            xaxis_title='Epochs',
            yaxis_title=metric.capitalize(),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def visualize_model_layers(self, model, model_name):
        """Create a visualization of the model's layer structure.
        
        Args:
            model: Keras model to visualize
            model_name (str): Name of the model
            
        Returns:
            str: Path to the generated visualization
        """
        # Extract layer information
        layers_info = []
        for i, layer in enumerate(model.layers):
            layers_info.append({
                'layer_number': i,
                'layer_name': layer.name,
                'layer_type': layer.__class__.__name__,
                'parameters': layer.count_params(),
                'input_shape': str(layer.input_shape),
                'output_shape': str(layer.output_shape)
            })
        
        # Create DataFrame
        df = pd.DataFrame(layers_info)
        
        # Create visualization using Plotly
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title=f"{model_name} Layer Structure"
        )
        
        if self.output_dir:
            output_path = os.path.join(self.output_dir, f"{model_name}_layers.html")
            fig.write_html(output_path)
            return output_path
        
        return None 