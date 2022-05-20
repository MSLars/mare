local bert_model = "roberta-large";
local dim = 1024;
local emb_learn_rate = 5e-6;
local learn_rate = 1e-4;

{
    dataset_reader: {
        type: "span_re",
        max_span_width: 10,
        tag_label: "relation",
        token_indexers: {
            bert: {
                type: "pretrained_transformer_mismatched",
                max_length: 512,
                model_name: bert_model
            }
        }
    },
    model: {
        type: "span_ner_tagger",
        feedforward: {
            activations: "relu",
            dropout: 0.05,
            hidden_dims: dim,
            input_dim: dim,
            num_layers: 2
        },
        initializer: {
            regexes: [
                [
                    ".*ner_scorer.*weight.*",
                    {
                        type: "xavier_uniform"
                    }
                ],
                [
                    ".*span_extractor.*weight.*",
                    {
                        type: "xavier_uniform"
                    }
                ]
            ]
        },
        max_inner_range: 20,
        ner_threshold: 0.5,
        regularizer: {
            regexes: [
                [
                    ".*ner_scorer.*weight.*",
                    {
                        alpha: 0.007774563908924041,
                        type: "l2"
                    }
                ],
                [
                    ".*span_extractor.*weight.*",
                    {
                        alpha: 0.007774563908924041,
                        type: "l2"
                    }
                ]
            ]
        },
        span_extractor: {
            type: "self_attentive",
            input_dim: dim
        },
        text_field_embedder: {
            token_embedders: {
                bert: {
                    type: "pretrained_transformer_mismatched",
                    max_length: 512,
                    model_name: "roberta-large"
                }
            }
        }
    },
    train_data_path: "data/re_train.jsonl",
    validation_data_path: "data/re_dev.jsonl",
    test_data_path: "data/re_test.jsonl",
    trainer: {
        cuda_device: 1,
        grad_norm: 5,
        learning_rate_scheduler: {
            type: "reduce_on_plateau",
            factor: 0.5,
            mode: "max",
            patience: 5
        },
        num_epochs: 50,
        optimizer: {
            type: "adamw",
            lr: learn_rate,
            parameter_groups: [
                [
                    [
                        "_embedder"
                    ],
                    {
                        finetune: true,
                        lr: emb_learn_rate,
                        weight_decay: 0.01
                    }
                ]
            ],
            weight_decay: 0
        },
        patience: 10,
        validation_metric: "+relation_f1"
    },
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: 8
        }
    },
    evaluate_on_test: true
}