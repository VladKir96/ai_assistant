"""Persistent Telegram audio response preference, independent of conversation threads."""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chat_preferences",
        sa.Column("chat_id", sa.String(100), primary_key=True),
        sa.Column("audio_enabled", sa.Boolean(), nullable=False),
    )


def downgrade():
    op.drop_table("chat_preferences")
