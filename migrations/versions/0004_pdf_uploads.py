"""PDF upload retry identity and bounded MCP evidence."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("runs", sa.Column("pdf_digest", sa.String(64), nullable=True))
    op.add_column("runs", sa.Column("pdf_evidence", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("runs", "pdf_evidence")
    op.drop_column("runs", "pdf_digest")
