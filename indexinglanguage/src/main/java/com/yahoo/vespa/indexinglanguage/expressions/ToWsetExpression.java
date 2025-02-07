// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.indexinglanguage.expressions;

import com.yahoo.document.DataType;
import com.yahoo.document.WeightedSetDataType;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.WeightedSet;

/**
 * @author Simon Thoresen Hult
 */
public final class ToWsetExpression extends Expression {

    private final Boolean createIfNonExistent;
    private final Boolean removeIfZero;

    public ToWsetExpression(boolean createIfNonExistent, boolean removeIfZero) {
        super(UnresolvedDataType.INSTANCE);
        this.createIfNonExistent = createIfNonExistent;
        this.removeIfZero = removeIfZero;
    }

    public boolean getCreateIfNonExistent() { return createIfNonExistent; }

    public boolean getRemoveIfZero() { return removeIfZero; }

    @Override
    protected void doVerify(VerificationContext context) {
        context.setCurrentType(DataType.getWeightedSet(context.getCurrentType(), createIfNonExistent, removeIfZero));
    }

    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Override
    protected void doExecute(ExecutionContext context) {
        FieldValue input = context.getCurrentValue();
        DataType inputType = input.getDataType();

        WeightedSetDataType outputType = DataType.getWeightedSet(inputType, createIfNonExistent, removeIfZero);
        WeightedSet output = outputType.createFieldValue();
        output.add(input);

        context.setCurrentValue(output);
    }

    @Override
    public DataType createdOutputType() { return UnresolvedDataType.INSTANCE; }

    @Override
    public String toString() {
        return "to_wset" +
               (createIfNonExistent ? " create_if_non_existent" : "") +
               (removeIfZero ? " remove_if_zero" : "");
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ToWsetExpression rhs)) return false;

        if (createIfNonExistent != rhs.createIfNonExistent) return false;
        if (removeIfZero != rhs.removeIfZero) return false;
        return true;
    }

    @Override
    public int hashCode() {
        return getClass().hashCode() + createIfNonExistent.hashCode() + removeIfZero.hashCode();
    }

}
