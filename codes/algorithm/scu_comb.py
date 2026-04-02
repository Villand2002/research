from typing import Any, Dict, List, Optional, Tuple  # typing helpers

import networkx as nx  # max-flow implementation

from codes.algorithm.scu import SCUSolver  # base SCU solver


class SCUcomb(SCUSolver):  # SCU solver with compact flow network
    def __init__(self, agents, categories, capacities, priorities, precedence, beneficial):
        super().__init__(agents, categories, capacities, priorities, precedence, beneficial)  # init base
        self._build_groups()  # precompute eligibility groups
        self._total_capacity = sum(self.capacities.values())  # sum of all category capacities
        self._beneficial_capacity = sum(  # sum of beneficial capacities
            self.capacities[c_id] for c_id in self.category_ids if c_id in self.beneficial_ids
        )
        self._non_beneficial_capacity = self._total_capacity - self._beneficial_capacity  # rest

    def solve(self) -> Dict[Any, Any]:
        """
        Compute the SCU matching using a compact flow network with lower bounds.
        """
        b = self._compute_max_beneficial_compact()  # step 1: max beneficial
        m = self._compute_max_cardinality_compact(b)  # step 2: max total

        assigned: Dict[Any, Any] = {}  # final assignment map
        assigned_counts: Dict[Tuple[Any, Any], int] = {}  # group-category fixed counts

        for cat_id in self.precedence:  # iterate categories by precedence
            if cat_id not in self.priorities:  # skip if no priority list
                continue  # nothing to do

            for ag_id in self.priorities[cat_id]:  # iterate agents by priority
                if ag_id in assigned:  # skip already assigned agents
                    continue  # already matched

                if not self._can_assign_compact(  # feasibility check with lower bounds
                    ag_id, cat_id, assigned_counts, b, m
                ):
                    continue  # cannot fix this assignment

                assigned[ag_id] = cat_id  # commit assignment
                group_id = self._agent_group[ag_id]  # lookup group id
                assigned_counts[(group_id, cat_id)] = assigned_counts.get(  # update group count
                    (group_id, cat_id), 0
                ) + 1

                if list(assigned.values()).count(cat_id) >= self.capacities[cat_id]:  # capacity reached
                    break  # move to next category

        return assigned  # return final allocation

    def _build_groups(self) -> None:
        """
        Group agents by their eligibility set to build a compact flow network.
        """
        groups: Dict[Tuple[Any, ...], List[Any]] = {}  # eligibility signature -> agents
        for ag_id in self.agent_ids:  # iterate all agents
            eligible = tuple(  # ordered eligibility list for key
                c_id for c_id in self.category_ids if ag_id in self.priorities.get(c_id, [])
            )
            groups.setdefault(eligible, []).append(ag_id)  # add agent to group

        self._groups: Dict[str, Dict[str, Any]] = {}  # group metadata
        self._agent_group: Dict[Any, str] = {}  # agent -> group id
        for idx, (eligible, agents) in enumerate(groups.items()):  # enumerate groups
            group_id = f"group_{idx}"  # stable group id
            self._groups[group_id] = {  # record group info
                "eligible": list(eligible),
                "size": len(agents),
            }
            for ag_id in agents:  # map each agent to group
                self._agent_group[ag_id] = group_id  # store mapping

    def _build_compact_graph(self, c0_capacity: int) -> Tuple[nx.DiGraph, str, str]:
        """
        Construct the compact flow network with the given upper bound on (C0, t).
        """
        G = nx.DiGraph()  # directed flow graph
        S, T = "source", "sink"  # terminals
        C_star, C_zero = "C_star", "C_zero"  # super category nodes

        for group_id, info in self._groups.items():  # add group nodes
            G.add_edge(S, group_id, capacity=info["size"])  # supply from source
            for c_id in info["eligible"]:  # connect to eligible categories
                G.add_edge(group_id, c_id, capacity=info["size"])  # group -> category

        for c_id in self.category_ids:  # add category -> super nodes
            target = C_star if c_id in self.beneficial_ids else C_zero  # choose super node
            G.add_edge(c_id, target, capacity=self.capacities[c_id])  # category capacity

        G.add_edge(C_star, T, capacity=self._beneficial_capacity)  # beneficial total cap
        G.add_edge(C_zero, T, capacity=c0_capacity)  # open total cap

        return G, S, T  # return graph and terminals

    def _compute_max_beneficial_compact(self) -> int:
        """
        Step 1: maximum flow when only beneficial categories are allowed.
        """
        G, S, T = self._build_compact_graph(c0_capacity=0)  # disable C0 capacity
        val, _ = nx.maximum_flow(G, S, T)  # compute max flow
        return val  # maximum beneficial matches

    def _compute_max_cardinality_compact(self, b: int) -> int:
        """
        Step 2: maximum total flow with at least b beneficial matches.
        """
        G, S, T = self._build_compact_graph(c0_capacity=self._non_beneficial_capacity)  # full graph
        lower_bounds = {( "C_star", T ): b}  # require b beneficial matches
        result = self._max_flow_with_lower_bounds(G, lower_bounds, S, T)  # run bounded max flow
        if result is None:  # infeasible case
            return 0  # return zero matches
        return result  # maximum total matches

    def _can_assign_compact(
        self,
        ag_id,
        cat_id,
        assigned_counts: Dict[Tuple[Any, Any], int],
        b: int,
        m: int,
    ) -> bool:
        """
        Step 3 feasibility check using the compact network with lower bounds.
        """
        group_id = self._agent_group[ag_id]  # group of current agent
        if cat_id not in self._groups[group_id]["eligible"]:  # eligibility check
            return False  # cannot assign

        G, S, T = self._build_compact_graph(c0_capacity=self._non_beneficial_capacity)  # base graph
        assigned_by_group: Dict[Any, int] = {}  # group -> assigned count
        assigned_by_category: Dict[Any, int] = {}  # category -> assigned count
        for (g_id, c_id), count in assigned_counts.items():  # apply upper-bound reductions
            if count <= 0:  # skip non-positive counts
                continue  # nothing to apply
            assigned_by_group[g_id] = assigned_by_group.get(g_id, 0) + count  # sum per group
            assigned_by_category[c_id] = assigned_by_category.get(c_id, 0) + count  # sum per category
            if G.has_edge(g_id, c_id):  # shrink group->category capacity
                G[g_id][c_id]["capacity"] = max(0, G[g_id][c_id]["capacity"] - count)

        for g_id, count in assigned_by_group.items():  # shrink source->group capacity
            if G.has_edge(S, g_id):
                G[S][g_id]["capacity"] = max(0, G[S][g_id]["capacity"] - count)

        for c_id, count in assigned_by_category.items():  # shrink category->super capacity
            if c_id in self.beneficial_ids:  # pick super node
                target = "C_star"
            else:
                target = "C_zero"
            if G.has_edge(c_id, target):
                G[c_id][target]["capacity"] = max(0, G[c_id][target]["capacity"] - count)

        assigned_beneficial = sum(
            count for c_id, count in assigned_by_category.items() if c_id in self.beneficial_ids
        )
        assigned_non_beneficial = sum(
            count for c_id, count in assigned_by_category.items() if c_id not in self.beneficial_ids
        )
        remaining_beneficial = b - assigned_beneficial
        remaining_non_beneficial = (m - b) - assigned_non_beneficial
        if remaining_beneficial < 0 or remaining_non_beneficial < 0:
            return False

        lower_bounds: Dict[Tuple[Any, Any], int] = {  # set required remaining totals
            ("C_star", T): remaining_beneficial,
            ("C_zero", T): remaining_non_beneficial,
        }
        lower_bounds[(group_id, cat_id)] = lower_bounds.get((group_id, cat_id), 0) + 1  # try assign
        return self._has_feasible_flow(G, lower_bounds, S, T)  # check feasibility

    def _apply_lower_bounds(
        self, G: nx.DiGraph, lower_bounds: Dict[Tuple[Any, Any], int]
    ) -> Tuple[nx.DiGraph, Dict[Any, int], bool]:
        """
        Convert lower bounds into node demands and residual capacities.
        """
        H = nx.DiGraph()  # residual capacity graph
        H.add_nodes_from(G.nodes)  # copy nodes
        demand: Dict[Any, int] = {node: 0 for node in G.nodes}  # demand balance per node

        for u, v, data in G.edges(data=True):  # transform each edge
            capacity = data.get("capacity", 0)  # original capacity
            lower = lower_bounds.get((u, v), 0)  # lower bound
            if lower > capacity:  # infeasible lower bound
                return H, demand, False  # fail fast
            H.add_edge(u, v, capacity=capacity - lower)  # residual capacity
            demand[u] -= lower  # supply decreases
            demand[v] += lower  # demand increases

        return H, demand, True  # return transformed graph

    def _has_feasible_flow(
        self,
        G: nx.DiGraph,
        lower_bounds: Dict[Tuple[Any, Any], int],
        source: str,
        sink: str,
    ) -> bool:
        """
        Check feasibility of a circulation with lower bounds.
        """
        H, demand, ok = self._apply_lower_bounds(G, lower_bounds)  # reduce lower bounds
        if not ok:  # infeasible immediately
            return False  # no feasible flow

        super_source = "_super_source"  # super source for demands
        super_sink = "_super_sink"  # super sink for supplies
        H.add_node(super_source)  # add super source
        H.add_node(super_sink)  # add super sink

        total_demand = 0  # sum of positive demands
        for node, dval in demand.items():  # connect demand edges
            if dval > 0:  # node needs inflow
                H.add_edge(super_source, node, capacity=dval)  # supply from super source
                total_demand += dval  # accumulate demand
            elif dval < 0:  # node has excess outflow
                H.add_edge(node, super_sink, capacity=-dval)  # send to super sink

        H.add_edge(sink, source, capacity=self._total_capacity + len(self.agent_ids))  # circulation edge
        flow_val, _ = nx.maximum_flow(H, super_source, super_sink)  # check feasibility
        return flow_val == total_demand  # feasible iff all demands satisfied

    def _max_flow_with_lower_bounds(
        self,
        G: nx.DiGraph,
        lower_bounds: Dict[Tuple[str, str], int],
        source: str,
        sink: str,
    ) -> Optional[int]:
        """
        Compute maximum flow with lower bounds via circulation reduction.
        """
        H, demand, ok = self._apply_lower_bounds(G, lower_bounds)  # convert bounds
        if not ok:  # infeasible bounds
            return None  # no feasible flow

        super_source = "_super_source"  # super source for demands
        super_sink = "_super_sink"  # super sink for supplies
        H.add_node(super_source)  # add super source
        H.add_node(super_sink)  # add super sink

        total_demand = 0  # sum of positive demands
        for node, dval in demand.items():  # connect demand edges
            if dval > 0:  # node requires inflow
                H.add_edge(super_source, node, capacity=dval)  # supply from super source
                total_demand += dval  # accumulate demand
            elif dval < 0:  # node has excess
                H.add_edge(node, super_sink, capacity=-dval)  # drain to super sink

        H.add_edge(sink, source, capacity=self._total_capacity + len(self.agent_ids))  # circulation edge
        flow_val, flow_dict = nx.maximum_flow(H, super_source, super_sink)  # satisfy demands
        if flow_val != total_demand:  # not all demands satisfied
            return None  # infeasible

        base_flow = flow_dict.get(sink, {}).get(source, 0)  # flow on back edge

        residual = nx.DiGraph()  # residual graph for extra flow
        for u, v, data in H.edges(data=True):  # build residual edges
            if u == sink and v == source:  # drop circulation edge for max-flow phase
                continue
            capacity = data["capacity"]  # edge capacity
            used = flow_dict.get(u, {}).get(v, 0)  # used flow
            remaining = capacity - used  # remaining capacity
            if remaining > 0:  # forward residual
                residual.add_edge(u, v, capacity=remaining)  # add forward edge
            if used > 0:  # backward residual
                residual.add_edge(v, u, capacity=used)  # add backward edge

        if residual.has_node(super_source):  # drop super source
            residual.remove_node(super_source)  # remove it
        if residual.has_node(super_sink):  # drop super sink
            residual.remove_node(super_sink)  # remove it

        extra_flow, _ = nx.maximum_flow(residual, source, sink)  # augment flow
        return base_flow + extra_flow  # total max flow
