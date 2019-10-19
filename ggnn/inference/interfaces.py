from abc import abstractmethod, ABC
from typing import List, Tuple, Dict

from autopandas.synthesis.model.featurization.io_featurizer import RelationGraph, RelationGraphEdgeType
from autopandas.synthesis.model.featurization.io_featurizer_visual import visualize_relation_graph, html_dataframe
from autopandas.synthesis.model.inference.interfaces import RelGraphInterface
from ggnn.models.base import BaseGGNN
from ggnn.models.sparse.base import SparseGGNN
from ggnn.models.sparse.seq.static_rnn import GGNNSeqStaticRNN


class GGNNInterface(RelGraphInterface, ABC):
    @abstractmethod
    def get_model(self):
        pass

    def init(self, model_dir):
        self.model: BaseGGNN = self.get_model()
        self.model.setup_inference(model_dir)

    def predict_graphs(self, graphs: List[Dict], with_confidence=True) -> List[List[Tuple[str, float]]]:
        return self.model.infer(graphs)


class SparseGGNNInterface(GGNNInterface):
    def get_model(self):
        return SparseGGNN()

    def debug_graph(self, graph: RelationGraph):
        model: SparseGGNN = self.model
        d_graph = graph.to_dict()
        result: List[Tuple[str, float]] = model.infer([d_graph])[0]
        print(result)
        attention = model.debug_attention(d_graph)

        #  Create the html report
        self.create_html_attention_report(graph, attention, result)

    @classmethod
    def create_html_attention_report(cls, graph, attention, preds):
        html = """
        <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        </head>
        """

        html += "<p>%s</p>" % (str(preds))

        #  Get the int to node mapping
        #  Works only because of insertion order preservation in python 3.6+
        int_to_node = {}
        for idx, node in enumerate(graph.nodes):
            int_to_node[idx] = node

        listener_template = """
        $("#%s").mouseover(function() {
            %s 
        });
        $("#%s").mouseout(function() {
            %s 
        });
        """

        def over_template(src_id, weight, attn):
            return """
            $("#%s").css({'border':'solid %dpx red'})
            $("#%s").attr('old', $("#%s").text())
            $("#%s").html("%.2f")
            """ % (src_id, weight, src_id, src_id, src_id, attn)

        def out_template(src_id):
            return """
            $("#%s").css({'border':''})
            $("#%s").html($("#%s").attr('old'))
            """ % (src_id, src_id, src_id)

        for (layer_idx, step), layer_attns in attention.items():
            html += "<hr><h3>Layer %d Time-step %d </h3>\n" % (layer_idx, step)
            html += "<h4> Inputs </h4>\n"
            for input_df_idx in range(len(graph.input_dfs)):
                input_df = graph.input_dfs[input_df_idx]
                idx = "%d-%d-I%d" % (layer_idx, step, input_df_idx)
                html += html_dataframe(input_df, idx, graph.options.COLUMN_NODES, graph.options.INDEX_NODES,
                                       bgcolor="lightgray", border="1", cellborder="2", cellspacing="10")[
                        1:-1] + "<br>\n"

            html += "<h4> Outputs </h4>\n"
            for output_df_idx in range(len(graph.output_dfs)):
                output_df = graph.output_dfs[output_df_idx]
                idx = "%d-%d-O%d" % (layer_idx, step, output_df_idx)
                html += html_dataframe(output_df, idx, graph.options.COLUMN_NODES, graph.options.INDEX_NODES,
                                       bgcolor="lightgray", border="1", cellborder="2", cellspacing="10")[
                        1:-1] + "<br>\n"

            #  Now add hovering logic for each of the nodes
            html += "<script>"
            for target, attns in layer_attns.items():
                target = int_to_node[target]
                tgt_el_id = "%d-%d-%s:%d.%d" % (layer_idx, step, target.dfindex, target.pos[0], target.pos[1])
                tgt_el_id = cls.jquery_escape(tgt_el_id)
                body_over = ""
                body_out = ""
                weight_map = {}
                mul = 1
                for i, val in enumerate(reversed(sorted(set(map(lambda x: x[1], attns))))):
                    weight_map[val] = mul * max(len(RelationGraphEdgeType), len(attns)) - mul * i + 1

                for src, attn in attns:
                    src = int_to_node[src]
                    src_el_id = "%d-%d-%s:%d.%d" % (layer_idx, step, src.dfindex, src.pos[0], src.pos[1])
                    src_el_id = cls.jquery_escape(src_el_id)
                    weight = weight_map[attn]
                    body_over += over_template(src_el_id, weight, attn)
                    body_out += out_template(src_el_id)

                html += listener_template % (tgt_el_id, body_over, tgt_el_id, body_out)

            html += "\n</script>"

        with open('debug.html', 'w') as f:
            print(html, file=f)

    @classmethod
    def jquery_escape(cls, tgt_el_id):
        return tgt_el_id.replace('.', '\\\\.').replace(':', '\\\\:')


class GGNNSeqStaticRNNInterface(GGNNInterface):
    def get_model(self):
        return GGNNSeqStaticRNN()

    def debug_graph(self, graph: RelationGraph):
        pass
