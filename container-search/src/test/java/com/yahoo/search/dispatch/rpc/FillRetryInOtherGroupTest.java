// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.search.dispatch.rpc;

import com.yahoo.compress.CompressionType;
import com.yahoo.container.protect.Error;
import com.yahoo.prelude.fastsearch.DocumentDatabase;
import com.yahoo.prelude.fastsearch.FastHit;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.dispatch.searchcluster.AvailabilityPolicy;
import com.yahoo.search.dispatch.searchcluster.Group;
import com.yahoo.search.dispatch.searchcluster.Node;
import com.yahoo.search.dispatch.searchcluster.SearchGroups;
import com.yahoo.search.dispatch.searchcluster.SearchGroupsImpl;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.Hit;
import com.yahoo.search.schema.DocumentSummary;
import com.yahoo.search.schema.Schema;
import com.yahoo.slime.BinaryFormat;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.function.IntFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the bounded retry of empty docsums in a single other group, used when there are too many empty
 * docsums to retry on every node - e.g. when a whole bucket moved away while a content node was replaced.
 *
 * @author djocas
 */
class FillRetryInOtherGroupTest {

    private static final int HITS = 20; // well above the default retry limit of 10

    @Test
    void too_many_empty_docsums_are_retried_on_a_single_other_group() {
        var nodes = new Fixture().group(0).group(1).group(2)
                                 .replying(0, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::docsums)
                                 .replying(2, FillRetryInOtherGroupTest::docsums);
        Result result = nodes.fill(new Query("?query=test"), HITS, 0);

        assertNull(result.hits().getError(), "all hits filled by the retry");
        assertAllFilled(result);
        assertEquals(1, nodes.asked.get(0).size(), "the first wave");
        int retriesOn1 = nodes.asked.get(1).size();
        int retriesOn2 = nodes.asked.get(2).size();
        assertEquals(1, retriesOn1 + retriesOn2, "retried on exactly one other group, not on every node");
    }

    @Test
    void a_group_without_sufficient_coverage_is_not_chosen() {
        var nodes = new Fixture().group(0).group(1).group(2)
                                 .withoutSufficientCoverage(1)
                                 .replying(0, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::docsums)
                                 .replying(2, FillRetryInOtherGroupTest::docsums);
        Result result = nodes.fill(new Query("?query=test"), HITS, 0);

        assertAllFilled(result);
        assertEquals(0, nodes.asked.get(1).size());
        assertEquals(1, nodes.asked.get(2).size());
    }

    @Test
    void all_working_nodes_of_a_multi_node_group_are_asked() {
        var nodes = new Fixture().group(0, 0, 1).group(1, 2, 3)
                                 .replying(0, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(2, FillRetryInOtherGroupTest::docsums)
                                 .replying(3, FillRetryInOtherGroupTest::emptyDocsums);
        Result result = nodes.fill(new Query("?query=test"), HITS, 0, 1);

        assertNull(result.hits().getError());
        assertAllFilled(result);
        assertEquals(List.of(HITS), nodes.asked.get(2), "every skipped hit, as we do not know which node holds it");
        assertEquals(List.of(HITS), nodes.asked.get(3));
    }

    @Test
    void no_retry_when_disabled_by_the_query() {
        var nodes = new Fixture().group(0).group(1)
                                 .replying(0, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::docsums);
        Result result = nodes.fill(new Query("?query=test&dispatch.docsumRetryOtherGroup=false"), HITS, 0);

        assertEmptyDocsumsError(result);
        assertEquals(0, nodes.asked.get(1).size());
    }

    @Test
    void no_retry_when_a_content_node_ran_out_of_time() {
        var nodes = new Fixture().group(0).group(1)
                                 .replying(0, FillRetryInOtherGroupTest::timedOutDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::docsums);
        Result result = nodes.fill(new Query("?query=test"), HITS, 0);

        assertEquals(0, nodes.asked.get(1).size(), "retrying would only add load to an overloaded system");
        assertTrue(result.hits().getErrorHit().errors().stream().anyMatch(e -> e.getCode() == Error.TIMEOUT.code));
    }

    @Test
    void no_retry_with_a_single_group() {
        var nodes = new Fixture().group(0, 0, 1)
                                 .replying(0, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::docsums);
        Result result = nodes.fill(new Query("?query=test"), HITS, 0);

        assertEmptyDocsumsError(result);
        assertEquals(0, nodes.asked.get(1).size());
    }

    @Test
    void the_retry_still_reports_hits_no_group_could_fill() {
        var nodes = new Fixture().group(0).group(1)
                                 .replying(0, FillRetryInOtherGroupTest::emptyDocsums)
                                 .replying(1, FillRetryInOtherGroupTest::emptyDocsums);
        Result result = nodes.fill(new Query("?query=test"), HITS, 0);

        assertEquals(1, nodes.asked.get(1).size());
        assertEmptyDocsumsError(result);
    }

    // ---- fixture ------------------------------------------------------------------------------------------

    private static void assertAllFilled(Result result) {
        for (Hit hit : (Iterable<Hit>) result.hits()::unorderedDeepIterator) {
            if (hit instanceof FastHit) assertNotNull(hit.getField("title"), "filled: " + hit);
        }
    }

    private static void assertEmptyDocsumsError(Result result) {
        ErrorMessage error = result.hits().getError();
        assertNotNull(error);
        assertEquals(Error.EMPTY_DOCUMENTS.code, error.getCode());
    }

    private static class Fixture {

        private final Map<Integer, Group> groups = new LinkedHashMap<>();
        private final Map<Integer, IntFunction<Client.ProtobufResponse>> replies = new HashMap<>();
        /** The number of hits each node was asked for, one entry per request */
        final Map<Integer, List<Integer>> asked = new HashMap<>();

        /** Adds a group with the given node keys, or a single node with the group id as key if none are given */
        Fixture group(int id, int... nodeKeys) {
            if (nodeKeys.length == 0) nodeKeys = new int[] { id };
            List<Node> nodes = new ArrayList<>();
            for (int key : nodeKeys) {
                Node node = new Node("test", key, "host" + key, id, true);
                node.setWorking(true);
                nodes.add(node);
                asked.put(key, new ArrayList<>());
            }
            Group group = new Group(id, nodes);
            group.setHasSufficientCoverage(true);
            groups.put(id, group);
            return this;
        }

        Fixture withoutSufficientCoverage(int groupId) {
            groups.get(groupId).setHasSufficientCoverage(false);
            return this;
        }

        Fixture replying(int nodeKey, IntFunction<Client.ProtobufResponse> reply) {
            replies.put(nodeKey, reply);
            return this;
        }

        /** Fills a result with the given number of hits, spread over the given nodes */
        Result fill(Query query, int hits, int... nodeKeys) {
            Result result = new Result(query);
            for (int i = 0; i < hits; i++) {
                FastHit hit = new FastHit(new byte[12], 1.0, OptionalInt.empty(), 0, nodeKeys[i % nodeKeys.length]);
                hit.setFillable();
                result.hits().add(hit);
            }
            invoker().fill(result, "default");
            return result;
        }

        private RpcProtobufFillInvoker invoker() {
            Map<Integer, Client.NodeConnection> connections = new HashMap<>();
            replies.forEach((key, reply) -> connections.put(key, connection(key, reply)));
            SearchGroups searchGroups = new SearchGroupsImpl(new AvailabilityPolicy(true, 100), groups);
            Schema schema = new Schema.Builder("test")
                    .add(new DocumentSummary.Builder("default").addField("title", "string").build())
                    .build();
            return new RpcProtobufFillInvoker(new RpcResourcePool(connections), searchGroups, new CompressService(),
                                              new DocumentDatabase(schema), "container.0",
                                              RpcProtobufFillInvoker.DecodePolicy.EAGER, false);
        }

        private Client.NodeConnection connection(int key, IntFunction<Client.ProtobufResponse> reply) {
            return new Client.NodeConnection() {
                @Override
                public void request(String rpcMethod, CompressionType compression, int uncompressedLength,
                                    byte[] compressedPayload, Client.ResponseReceiver receiver, double timeoutSeconds) {
                    int numHits = numHitsIn(compression, uncompressedLength, compressedPayload);
                    asked.get(key).add(numHits);
                    receiver.receive(Client.ResponseOrError.fromResponse(reply.apply(numHits)));
                }
                @Override public void close() { }
            };
        }

        private static int numHitsIn(CompressionType compression, int uncompressedLength, byte[] payload) {
            try {
                byte[] bytes = new CompressService().decompress(new Client.ProtobufResponse(compression.getCode(),
                                                                                            uncompressedLength,
                                                                                            payload));
                return ai.vespa.searchlib.searchprotocol.protobuf.SearchProtocol.DocsumRequest.parseFrom(bytes).getGlobalIdsCount();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

    }

    private static Client.ProtobufResponse docsums(int n) {
        return reply(n, true, false);
    }

    /** How the content layer reports "I no longer hold this document": an entry without a docsum */
    private static Client.ProtobufResponse emptyDocsums(int n) {
        return reply(n, false, false);
    }

    /** How the content layer reports running out of time: fewer entries than requested, and a timeout error */
    private static Client.ProtobufResponse timedOutDocsums(int n) {
        return reply(n, false, true);
    }

    private static Client.ProtobufResponse reply(int n, boolean withDocsums, boolean timedOut) {
        Slime slime = new Slime();
        Cursor root = slime.setObject();
        Cursor docsums = root.setArray("docsums");
        if ( ! timedOut) {
            for (int i = 0; i < n; i++) {
                Cursor entry = docsums.addObject();
                if (withDocsums) entry.setObject("docsum").setString("title", "title " + i);
            }
        } else {
            Cursor error = root.setArray("errors").addObject();
            error.setString("type", "timeout");
            error.setString("message", "Timed out " + n + " summaries with 0us left.");
        }
        byte[] payload = ai.vespa.searchlib.searchprotocol.protobuf.SearchProtocol.DocsumReply.newBuilder()
                .setSlimeSummaries(com.google.protobuf.ByteString.copyFrom(BinaryFormat.encode(slime)))
                .build().toByteArray();
        return new Client.ProtobufResponse(CompressionType.NONE.getCode(), payload.length, payload);
    }

}
